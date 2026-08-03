#include <iostream>
#include <chrono>

static double total_conv_time = 0.0;
static double total_matmul_time = 0.0;
static std::chrono::high_resolution_clock::time_point start_time;

extern "C" {
    void _mlir_ciface_start_timer() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void _mlir_ciface_stop_timer_conv() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        total_conv_time += duration.count();
    }

    void _mlir_ciface_stop_timer_matmul() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        total_matmul_time += duration.count();
    }
}

// Clase para imprimir automáticamente al acabar
struct ProfilerPrinter {
    ~ProfilerPrinter() {
        std::cout << "\n====================================\n";
        std::cout << "      RESULTADOS DEL PROFILING      \n";
        std::cout << "====================================\n";
        std::cout << "Tiempo total en Convoluciones : " << total_conv_time << " ms\n";
        std::cout << "Tiempo total en MatMul        : " << total_matmul_time << " ms\n";
        std::cout << "====================================\n\n";
    }
} printer_instance;