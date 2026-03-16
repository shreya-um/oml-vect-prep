/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Conv.cpp - Lowering Convolution Op -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConvOpLowering : public OpConversionPattern<ONNXConvOp> {
  ONNXConvOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXConvOp::getOperationName());
  }

  bool enableParallel;

 
void convGEMM(ConversionPatternRewriter &rewriter, ONNXConvOp &convOp,
      ONNXConvOpAdaptor &operandAdaptor, ONNXConvOpShapeHelper &shapeHelper,
      MemRefType &memRefType, Value alloc) const {
    Operation *op = convOp.getOperation();
    Location loc = convOp.getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, SCFBuilder,
        MathBuilder, MemRefBuilder>
        create(rewriter, loc);

    auto inputOperand = operandAdaptor.getX();
    auto filterOperand = operandAdaptor.getW();
    auto biasOperand = operandAdaptor.getB();
    bool hasBias = !mlir::isa<NoneType>(biasOperand.getType());
    int64_t groupNum = convOp.getGroup();
    IndexExpr G = LitIE(groupNum);

    // Bounds for output sizes: [N x CO x HO x WO]:
    // where N is Batch Size,
    // where CO (or M) is Channel Out (multiple of group num)
    // and where HO & WO are spacial dimensions of the output.

    // shapeHelper -> [N, C, H, W]
    // Output: OH -> Output Height
    // Output: OW -> Output Width
    IndexExpr N = shapeHelper.getOutputDims()[0];
    IndexExpr CO = shapeHelper.getOutputDims()[1];
    IndexExpr COPerGroup = CO.ceilDiv(G);
    IndexExpr OH = shapeHelper.getOutputDims()[2];
    IndexExpr OW = shapeHelper.getOutputDims()[3];
    
    auto bodyFunction = [&](ValueRange outerIndices) {

      // Bounds for input image X: [N x CI x HI x WI]:
      // where N is Batch Size,
      // where CI (or C) is Channel In (multiple of group num),
      // and where HI & WI are spacial dimensions of the input image.
      SymbolIndexExpr HI = create.krnlIE.getShapeAsSymbol(inputOperand, 2);
      SymbolIndexExpr WI = create.krnlIE.getShapeAsSymbol(inputOperand, 3);

      // Bounds for kernel/filter W: [CO x CIPerGroup x KH x KW]:
      // where CO (or M) is Channel Out,
      // where CIPerGroup (or C/G) is number of channel in per group,
      // and where KH x KW are the kernel / filter size (e.g. 3x3, 1x1).

      // filterOperand -> [M, C, KH, KW]
      // Kernel: KH -> Kernel Height
      // Kernel: KW -> Kernel Width
      SymbolIndexExpr KH = create.krnlIE.getShapeAsSymbol(filterOperand, 2);
      SymbolIndexExpr KW = create.krnlIE.getShapeAsSymbol(filterOperand, 3);

      // Channel per group, needed to determine the dataCol columns
      IndexExpr CIPerGroup = create.krnlIE.getShapeAsSymbol(filterOperand, 1);

      // dataCol rows and columns calculation
      IndexExpr dataColRows = CIPerGroup * KH * KW;
      IndexExpr dataColCols = OH * OW;

      // Obtain data type and allocate memory for dataCol
      Type elementType = memRefType.getElementType();
      // We create a 2D matrix to let the compiler use matrix optimizations
      // Instead of copying the 1D logic from the pseudo-code
      MemRefType dataColType = MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, elementType);
      Value dataCol = create.mem.alignedAlloc(dataColType, {dataColRows, dataColCols});

      // We need 3 nested loops
      ValueRange im2colLoops = create.krnl.defineLoops(3);
      
      // We define the bounds, lower and upper
      IndexExpr iZero = LitIE(0);
      SmallVector<IndexExpr, 3> im2colLbs = {iZero, iZero, iZero};
      SmallVector<IndexExpr, 3> im2colUbs = {dataColRows, OH, OW};

      // im2col

      create.krnl.iterateIE(im2colLoops, im2colLoops, im2colLbs, im2colUbs,
          [&](const KrnlBuilder &createKrnl, ValueRange im2colIndices){

            // Inner loop

            MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder, SCFBuilder>
                create(createKrnl);
            IndexExprScope innerScope(createKrnl); // esto se suele poner antes

            // Loop variables
            DimIndexExpr c(im2colIndices[0]);
            DimIndexExpr h(im2colIndices[1]);
            DimIndexExpr w(im2colIndices[2]);

            LiteralIndexExpr S_H(shapeHelper.strides[0]);
            LiteralIndexExpr S_W(shapeHelper.strides[1]);
            SymbolIndexExpr P_H(shapeHelper.pads[0]); // Top pad
            SymbolIndexExpr P_W(shapeHelper.pads[1]); // Left pad

            // Should be in outer loop
            IndexExpr w_offset = c % KW;
            IndexExpr h_offset = (c.floorDiv(KW)) % KH;
            IndexExpr c_im = c.floorDiv(KH * KW);

            IndexExpr g_index = DimIE(outerIndices[1]);
            IndexExpr absolute_c_im = g_index * DimIE(CIPerGroup) + c_im;

            IndexExpr row = h_offset + h * S_H - P_H;
            IndexExpr col = w_offset + w * S_W - P_W;

            // We don't need this (it's for the 1D array)
  //          IndexExpr col_index = (c * OH + h) * OW + w;
            // This is the actual col_index for this 2D matrix
            IndexExpr col_index = (h * OW) + w;

            // Create conditions for if statement
            Value zeroIdx = create.math.constantIndex(0);
            Value rowGEzero = create.math.sge(row.getValue(), zeroIdx);// row >= 0
            Value colGEzero = create.math.sge(col.getValue(), zeroIdx); // col >= 0
            Value rowLTH = create.math.slt(row.getValue(), HI.getValue());    // row < H
            Value colLTW = create.math.slt(col.getValue(), WI.getValue());    // col < W

            Value and1 = create.math.andi(rowGEzero, colGEzero);
            Value and2 = create.math.andi(rowLTH, colLTW);
            Value isValidPixel = create.math.andi(and1, and2);

            // Create the if statement
            create.scf.ifThenElse(isValidPixel, 
                [&](const SCFBuilder &createSCF){
                  // TRUE
                  // We create the index vector, load it from inputOperand and yield it
                  SmallVector<IndexExpr, 4> inputIndices = {DimIE(outerIndices[0]), absolute_c_im, row, col};
                  Value val = create.krnl.loadIE(inputOperand, inputIndices);
                  create.krnl.storeIE(val, dataCol, {c, col_index});
                }, 
                [&](const SCFBuilder &createSCF){
                  // FALSE
                  Value zeroVal = create.math.constant(elementType, 0);
                  create.krnl.storeIE(zeroVal, dataCol, {c, col_index});
                });
          });

  // GEMM

      IndexExpr n_limit = DimIE(COPerGroup);  // Rows of A
      IndexExpr m_limit = OH * OW;            // Columns of B (how many pixels in the output)
      IndexExpr p_limit = dataColRows;        // Cols of A and Rows of B (kernel size)

      ValueRange gemmLoops = create.krnl.defineLoops(2);
      SmallVector<IndexExpr, 2> gemmLbs = {LitIE(0), LitIE(0)};
      SmallVector<IndexExpr, 2> gemmUbs = {n_limit, m_limit};

      // i j loops
      create.krnl.iterateIE(gemmLoops, gemmLoops, gemmLbs, gemmUbs,
          [&](const KrnlBuilder &createKrnl, ValueRange gemmIndices){

            // i -> A rows
            // j -> B column
            DimIndexExpr i(gemmIndices[0]);
            DimIndexExpr j(gemmIndices[1]);

            MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder>
              create(createKrnl);

            // In Convolution alpha is always 1 and beta is always 0
            // Bounds for output sizes: [N x CO x C_oh x C_ow]
            // Matrix C indices
            IndexExpr C_ow = j % OW;
            IndexExpr C_oh = j.floorDiv(OW);

            // To calculate the output channel we need to do g * n + i
            // outerIndices [0] -> N (number of output channels per group)
            // outerIndices [1] -> g
            IndexExpr outputChannel = DimIE(outerIndices[1]) * n_limit + i;
            SmallVector<IndexExpr, 4> C_indices = {DimIE(outerIndices[0]), outputChannel, C_oh, C_ow};

            // Here we could initialize directly with bias if needed
            Value initVal;
            if(hasBias){
              initVal = create.krnl.loadIE(biasOperand, {outputChannel});
            }
            else{
              initVal = create.math.constant(elementType, 0);
            }

            create.krnl.storeIE(initVal, alloc, C_indices);

            // k loop
            ValueRange kLoop = create.krnl.defineLoops(1);
            SmallVector<IndexExpr, 1> kLbs = {LitIE(0)};
            SmallVector<IndexExpr, 1> kUbs = {p_limit};
            create.krnl.iterateIE(kLoop, kLoop, kLbs, kUbs,
                    [&](const KrnlBuilder &createKrnl, ValueRange kIndices){
                      MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder>
                        createK(createKrnl);
                      
                      DimIndexExpr k(kIndices[0]);

                      // We calculate A indexes
                      IndexExpr k_kw = k % KW;
                      IndexExpr k_kh = k.floorDiv(KW) % KH;
                      IndexExpr k_c = k.floorDiv(KH * KW);
                      // We use outputChannel to know the filter we are in
                      SmallVector<IndexExpr, 4> A_dimensions = {outputChannel, k_c, k_kh, k_kw};

                      // We calculate B indexes
                      SmallVector<IndexExpr, 2> B_dimensions = {k, j};
                      
                      // Load values from A and B
                      Value aVal = createK.krnl.loadIE(filterOperand, A_dimensions);
                      Value bVal = createK.krnl.loadIE(dataCol, B_dimensions);

                      // Multiply A * B
                      Value mult = createK.math.mul(aVal, bVal);

                      // C[i][j] += mult
                      Value currentVal = createK.krnl.loadIE(alloc, C_indices);
                      Value newVal = createK.math.add(currentVal, mult);
                      createK.krnl.storeIE(newVal, alloc, C_indices);
                    });
          });
    };


    IndexExpr iZero = LitIE(0);

    SmallVector<IndexExpr, 3> outerLbs = {iZero, iZero, iZero};
    SmallVector<IndexExpr, 3> outerUbs = {N, G, COPerGroup};
    

    ValueRange outerLoops = create.krnl.defineLoops(3);
    if (enableParallel)
      tryCreateKrnlParallel(
          create.krnl, op, "conv", outerLoops, outerLbs, outerUbs, 0, 1);
    create.krnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
        [&](const KrnlBuilder &create, ValueRange outerIndices) {
          bodyFunction(outerIndices);
        });
  }

  LogicalResult matchAndRewrite(ONNXConvOp convOp, ONNXConvOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = convOp.getOperation();
    Location loc = ONNXLoc<ONNXConvOp>(op);
    ValueRange operands = adaptor.getOperands();

    // Get shape.
    MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
        rewriter, loc);

    ONNXConvOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Insert allocation for the result of this operation.
    Value alloc = allocForONNXOp<ONNXConvOp>(
        convOp, rewriter, typeConverter, shapeHelper)[0];
    MemRefType memRefType = mlir::cast<MemRefType>(alloc.getType());
    convGEMM(rewriter, convOp, adaptor, shapeHelper, memRefType, alloc);

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel,
    std::string opsForCall) {
  patterns.insert<ONNXConvOpToCall>(typeConverter, ctx, opsForCall);
  patterns.insert<ONNXConvOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
