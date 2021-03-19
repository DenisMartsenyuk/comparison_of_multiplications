//
// Created by Денис Марценюк on 19.03.2021.
//

#ifndef COMPARISON_OF_MULTIPLICATIONS_MULTIPLICATIONMATRICESGPU_H
#define COMPARISON_OF_MULTIPLICATIONS_MULTIPLICATIONMATRICESGPU_H

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "libraries/cl2.hpp"
#include <fstream>

class MultiplicationMatricesGPU {
private:
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::NDRange workItems;
    cl::NDRange workGroup;
    cl::Kernel kernel;
    cl::CommandQueue queue;
    cl::Buffer bufferA;
    cl::Buffer bufferB;
    cl::Buffer bufferResult;

    std::string readKernel(std::string path);
    void createBuffersAndQueue(int rows, int columns, int general_size);

public:
    int init(int deviceNumber);
    void setProgram(std::string path);
    void setWorkGroupAndWorkItems(int rowsWorkGroup, int columnsWorkGroup, int rowsMatrix, int columnsMatrix);
    void setKernel(std::string nameKernel);
    void setArgs(float *matrixA, float *matrixB, int rows, int columns, int generalSize);
    void executeKernel();
    void getResult(float *matrixResult, int rows, int columns);
};


#endif //COMPARISON_OF_MULTIPLICATIONS_MULTIPLICATIONMATRICESGPU_H
