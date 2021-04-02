#include <iostream>
#include "matrices/MatrixOperations.h"
#include "config.h"
#include "gpu/MultiplicationMatricesGPU.h"

template <class T>
double test(MultiplicationMatricesGPU *multiplicationMatricesGpu, std::pair<std::string, std::string> *kernel, T *a, T *b, T *resultReference, T *resultKernel) {
    multiplicationMatricesGpu->setKernel(kernel->second);
    multiplicationMatricesGpu->setArgs(a, b, ROWS, COLUMNS, GENERAL_SIZE);
    multiplicationMatricesGpu->executeKernel();
    multiplicationMatricesGpu->getResult(resultKernel, ROWS, COLUMNS);
    if (MatrixOperations::compareMatrices(resultReference, resultKernel, ROWS, COLUMNS)) {
        return multiplicationMatricesGpu->getExecutionTime();
    }
    return -1.0;
}

template <class T>
void createTestData(MultiplicationMatricesGPU *multiplicationMatricesGpu, std::string kernel, T *a, T *b, T *resultReference) {
    MatrixOperations::generateMatrix(a, ROWS, GENERAL_SIZE);
    MatrixOperations::generateMatrix(b, GENERAL_SIZE, COLUMNS);
    multiplicationMatricesGpu->setKernel(kernel);
    multiplicationMatricesGpu->setArgs(a, b, ROWS, COLUMNS, GENERAL_SIZE);
    multiplicationMatricesGpu->executeKernel();
    multiplicationMatricesGpu->getResult(resultReference, ROWS, COLUMNS);
}

void printTestResult(int numberTest, std::vector<std::pair<std::string, std::vector<double>>> &resultTime) {
    std::cout << "Test " << numberTest << std::endl;
    for (int i = 0; i < resultTime.size(); ++i) {
//        std::cout << "Times " << resultTime[i].first << " kernel: ";
        for (int j = 0; j < resultTime[i].second.size(); ++j) {
            std::cout << resultTime[i].second[j] << " ";
        }
        std::cout << std::endl;
//        std::cout << "in milliseconds" << std::endl;
    }
    std::cout << std::endl;
}

int main() {

    std::vector<std::pair<std::string, std::string>> kernels;
    kernels.emplace_back(TYPE_KERNEL_1, NAME_KERNEL_1);
    kernels.emplace_back(TYPE_KERNEL_2, NAME_KERNEL_2);
    kernels.emplace_back(TYPE_KERNEL_3, NAME_KERNEL_3);
    kernels.emplace_back(TYPE_KERNEL_4, NAME_KERNEL_4);


    MultiplicationMatricesGPU multiplicationMatricesGpu = MultiplicationMatricesGPU();
    multiplicationMatricesGpu.init(DEVICE_NUMBER);
    multiplicationMatricesGpu.setProgram(PATH_TO_KERNEL_FILE);
    multiplicationMatricesGpu.setWorkGroupAndWorkItems(WORK_GROUP_ROWS, WORK_GROUP_COLUMNS, ROWS, COLUMNS);


    for (int i = 0; i < NUMBER_OF_MEASUREMENTS; ++i) {
        std::vector<std::pair<std::string, std::vector<double>>> resultTime;
        resultTime.emplace_back(NAME_KERNEL_1, std::vector<double> {});
        resultTime.emplace_back(NAME_KERNEL_2, std::vector<double> {});
        resultTime.emplace_back(NAME_KERNEL_3, std::vector<double> {});
        resultTime.emplace_back(NAME_KERNEL_4, std::vector<double> {});
        for (int j = 0; j < kernels.size(); ++j) {
            if (kernels[j].first == "double") {
                double *a = (double*)malloc(ROWS * GENERAL_SIZE * sizeof(double));
                double *b = (double*)malloc( GENERAL_SIZE * COLUMNS * sizeof(double));
                double *resultReference = (double*)malloc(ROWS * COLUMNS * sizeof(double));
                double *resultKernel = (double*)malloc(ROWS * COLUMNS * sizeof(double));
                createTestData(&multiplicationMatricesGpu, NAME_REFERENCE_KERNEL_DOUBLE, a, b, resultReference);
                for (int k = 0; k < NUMBER_OF_IDENTICAL_MEASUREMENTS; ++k) {
                    resultTime[j].second.push_back(test(&multiplicationMatricesGpu, &kernels[j], a, b, resultReference, resultKernel));
                }
            } else if(kernels[j].first == "float") {
                float *a = (float*)malloc(ROWS * GENERAL_SIZE * sizeof(float));
                float *b = (float*)malloc( GENERAL_SIZE * COLUMNS * sizeof(float));
                float *resultReference = (float*)malloc(ROWS * COLUMNS * sizeof(float));
                float *resultKernel = (float*)malloc(ROWS * COLUMNS * sizeof(float));
                createTestData(&multiplicationMatricesGpu, NAME_REFERENCE_KERNEL_FLOAT, a, b, resultReference);
                for (int k = 0; k < NUMBER_OF_IDENTICAL_MEASUREMENTS; ++k) {
                    resultTime[j].second.push_back(test(&multiplicationMatricesGpu, &kernels[j], a, b, resultReference, resultKernel));
                }
            }
        }
        printTestResult(i + 1, resultTime);
    }

    return 0;
}
