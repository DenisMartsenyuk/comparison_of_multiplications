#include <iostream>
#include <chrono>
#include "matrices/MatrixOperations.h"
#include "config.h"
#include "gpu/MultiplicationMatricesGPU.h"

int main() {

    float *a = (float*)malloc(ROWS * GENERAL_SIZE * sizeof(float));
    float *b = (float*)malloc( GENERAL_SIZE * COLUMNS * sizeof(float));
    float *resultCPU = (float*)malloc(ROWS * COLUMNS * sizeof(float));
    float *resultKernel1 = (float*)malloc(ROWS * COLUMNS * sizeof(float));
    float *resultKernel2 = (float*)malloc(ROWS * COLUMNS * sizeof(float));

    MultiplicationMatricesGPU multiplicationMatricesGpu = MultiplicationMatricesGPU();
    multiplicationMatricesGpu.init(DEVICE_NUMBER);
    multiplicationMatricesGpu.setProgram(PATH_TO_KERNEL_FILE);
    multiplicationMatricesGpu.setWorkGroupAndWorkItems(WORK_GROUP_ROWS, WORK_GROUP_COLUMNS, ROWS, COLUMNS);

    for (int i = 0; i < NUMBER_OF_MEASUREMENTS; ++i) {
        MatrixOperations::generateMatrix(a, ROWS, GENERAL_SIZE);
        MatrixOperations::generateMatrix(b, GENERAL_SIZE, COLUMNS);

        std::cout << "Test " << i + 1 << std::endl;

        auto startTime = std::chrono::high_resolution_clock::now();
        MatrixOperations::multiplicationMatrices(a, b, resultCPU, ROWS, COLUMNS, GENERAL_SIZE);
        auto finishTime = std::chrono::high_resolution_clock::now();
        auto executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(finishTime - startTime).count();
        double executionTimeMillis = executionTime / 1e6;
        std::cout << "CPU multiplication time: " << executionTimeMillis << " milliseconds" << std::endl;

        multiplicationMatricesGpu.setKernel(NAME_KERNEL_1);
        multiplicationMatricesGpu.setArgs(a, b, ROWS, COLUMNS, GENERAL_SIZE);
        startTime = std::chrono::high_resolution_clock::now();
        multiplicationMatricesGpu.executeKernel();
        finishTime = std::chrono::high_resolution_clock::now();
        executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(finishTime - startTime).count();
        executionTimeMillis = executionTime / 1e6;
        std::cout << "GPU kernel 1 multiplication time: " << executionTimeMillis << " milliseconds" << std::endl;
        multiplicationMatricesGpu.getResult(resultKernel1, ROWS, COLUMNS);
        if (MatrixOperations::compareMatrices(resultCPU, resultKernel1, ROWS, COLUMNS)) {
            std::cout << "Calculation in " << NAME_KERNEL_1 << " is correct." << std::endl;
        } else {
            std::cout << "Calculation in " << NAME_KERNEL_1 << " is incorrect." << std::endl;
        }

        multiplicationMatricesGpu.setKernel(NAME_KERNEL_2);
        multiplicationMatricesGpu.setArgs(a, b, ROWS, COLUMNS, GENERAL_SIZE);
        startTime = std::chrono::high_resolution_clock::now();
        multiplicationMatricesGpu.executeKernel();
        finishTime = std::chrono::high_resolution_clock::now();
        executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(finishTime - startTime).count();
        executionTimeMillis = executionTime / 1e6;
        std::cout << "GPU kernel 2 multiplication time: " << executionTimeMillis << " milliseconds" << std::endl;
        multiplicationMatricesGpu.getResult(resultKernel2, ROWS, COLUMNS);
        if (MatrixOperations::compareMatrices(resultCPU, resultKernel2, ROWS, COLUMNS)) {
            std::cout << "Calculation in " << NAME_KERNEL_2 << " is correct." << std::endl;
        } else {
            std::cout << "Calculation in " << NAME_KERNEL_2 << " is incorrect." << std::endl;
        }

        multiplicationMatricesGpu.setKernel(NAME_KERNEL_1);
        multiplicationMatricesGpu.setArgs(a, b, ROWS, COLUMNS, GENERAL_SIZE);
        startTime = std::chrono::high_resolution_clock::now();
        multiplicationMatricesGpu.executeKernel();
        finishTime = std::chrono::high_resolution_clock::now();
        executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(finishTime - startTime).count();
        executionTimeMillis = executionTime / 1e6;
        std::cout << "GPU kernel 1 multiplication time: " << executionTimeMillis << " milliseconds" << std::endl;
        multiplicationMatricesGpu.getResult(resultKernel1, ROWS, COLUMNS);
        if (MatrixOperations::compareMatrices(resultCPU, resultKernel1, ROWS, COLUMNS)) {
            std::cout << "Calculation in " << NAME_KERNEL_1 << " is correct." << std::endl;
        } else {
            std::cout << "Calculation in " << NAME_KERNEL_1 << " is incorrect." << std::endl;
        }

        std::cout << std::endl;
    }

    return 0;
}
