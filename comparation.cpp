#include <iostream>
#include "matrices/MatrixOperations.h"
#include "config.h"
#include "gpu/MultiplicationMatricesGPU.h"

template <class T>
double test(MultiplicationMatricesGPU *multiplicationMatricesGpu, std::pair<std::string, std::string> *kernel, T *a, T *b, T *resultCPU, T *resultKernel) {
    double gFlopsVariable = (ROWS * COLUMNS * GENERAL_SIZE * 2) * 1.0 / (1000 * 1000 * 1000);

    MatrixOperations::generateMatrix(a, ROWS, GENERAL_SIZE);
    MatrixOperations::generateMatrix(b, GENERAL_SIZE, COLUMNS);
    MatrixOperations::multiplicationMatrices(a, b, resultCPU, ROWS, COLUMNS, GENERAL_SIZE);

    multiplicationMatricesGpu->setKernel(kernel->second);
    multiplicationMatricesGpu->setArgs(a, b, ROWS, COLUMNS, GENERAL_SIZE);
    multiplicationMatricesGpu->executeKernel();
    std::cout << kernel->second << " multiplication time: " << multiplicationMatricesGpu->getExecutionTime() << " milliseconds" << std::endl;
    std::cout << kernel->second << " GFLOPS: " << gFlopsVariable / (multiplicationMatricesGpu->getExecutionTime() / 1000) << std::endl;
    multiplicationMatricesGpu->getResult(resultKernel, ROWS, COLUMNS);
    if (MatrixOperations::compareMatrices(resultCPU, resultKernel, ROWS, COLUMNS)) {
        std::cout << "Calculation in " << kernel->second << " is correct." << std::endl;
    } else {
        std::cout << "Calculation in " << kernel->second << " is correct." << std::endl;
    }

    return gFlopsVariable / (multiplicationMatricesGpu->getExecutionTime() / 1000);
}

int main() {

    std::vector<std::pair<std::string, std::string>> kernels;
    kernels.push_back(std::make_pair(TYPE_KERNEL_1, NAME_KERNEL_1));
    kernels.push_back(std::make_pair(TYPE_KERNEL_2, NAME_KERNEL_2));
    kernels.push_back(std::make_pair(TYPE_KERNEL_3, NAME_KERNEL_3));
    kernels.push_back(std::make_pair(TYPE_KERNEL_4, NAME_KERNEL_4));

    MultiplicationMatricesGPU multiplicationMatricesGpu = MultiplicationMatricesGPU();
    multiplicationMatricesGpu.init(DEVICE_NUMBER);
    multiplicationMatricesGpu.setProgram(PATH_TO_KERNEL_FILE);
    multiplicationMatricesGpu.setWorkGroupAndWorkItems(WORK_GROUP_ROWS, WORK_GROUP_COLUMNS, ROWS, COLUMNS);

    std::vector<std::pair<std::string, double>> resultGFLOPS;
    resultGFLOPS.push_back(std::make_pair(NAME_KERNEL_1, 0.0));
    resultGFLOPS.push_back(std::make_pair(NAME_KERNEL_2, 0.0));
    resultGFLOPS.push_back(std::make_pair(NAME_KERNEL_3, 0.0));
    resultGFLOPS.push_back(std::make_pair(NAME_KERNEL_4, 0.0));

    for (int i = 0; i < NUMBER_OF_MEASUREMENTS; ++i) {
        std::cout << "Test " << i + 1 << std::endl;

        for (int j = 0; j < kernels.size(); ++j) {
            if (kernels[j].first == "double") {
                double *a = (double*)malloc(ROWS * GENERAL_SIZE * sizeof(double));
                double *b = (double*)malloc( GENERAL_SIZE * COLUMNS * sizeof(double));
                double *resultCPU = (double*)malloc(ROWS * COLUMNS * sizeof(double));
                double *resultKernel = (double*)malloc(ROWS * COLUMNS * sizeof(double));
                resultGFLOPS[j].second += test(&multiplicationMatricesGpu, &kernels[j], a, b, resultCPU, resultKernel);
            } else if(kernels[j].first == "float") {
                float *a = (float*)malloc(ROWS * GENERAL_SIZE * sizeof(float));
                float *b = (float*)malloc( GENERAL_SIZE * COLUMNS * sizeof(float));
                float *resultCPU = (float*)malloc(ROWS * COLUMNS * sizeof(float));
                float *resultKernel = (float*)malloc(ROWS * COLUMNS * sizeof(float));
                resultGFLOPS[j].second += test(&multiplicationMatricesGpu, &kernels[j], a, b, resultCPU, resultKernel);
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << std::endl;
    }

    for (int i = 0; i < resultGFLOPS.size(); ++i) {
        std::cout << resultGFLOPS[i].first << " average value GFLOPS: " << resultGFLOPS[i].second / NUMBER_OF_MEASUREMENTS << std::endl;
        if (kernels[i].first == "float") {
            std::cout << "Percentage of theoretical performance: " << THEORETICAL_PERFORMANCE_FLOAT / 100.0 * resultGFLOPS[i].second / NUMBER_OF_MEASUREMENTS  << " %" << std::endl;
        } else if (kernels[i].first == "double") {
            std::cout << "Percentage of theoretical performance: " << THEORETICAL_PERFORMANCE_DOUBLE / 100.0 * resultGFLOPS[i].second / NUMBER_OF_MEASUREMENTS  << " %" << std::endl;
        }
    }

    return 0;
}
