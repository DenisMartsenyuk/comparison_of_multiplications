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
    cl::Event event;

    std::string readKernel(std::string path);

    template<class T>
            void createBuffersAndQueue(int rows, int columns, int general_size);

public:
    int init(int deviceNumber);
    void setProgram(std::string path);
    void setWorkGroupAndWorkItems(int rowsWorkGroup, int columnsWorkGroup, int rowsMatrix, int columnsMatrix);
    void setKernel(std::string nameKernel);

    template<class T>
            void setArgs(T *matrixA, T *matrixB, int rows, int columns, int generalSize);

    void executeKernel();
    double getExecutionTime();

    template<class T>
            void getResult(T *matrixResult, int rows, int columns);
};

template<class T>
void MultiplicationMatricesGPU::createBuffersAndQueue(int rows, int columns, int generalSize) {
    bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * rows * generalSize);
    bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * generalSize * columns);
    bufferResult = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * rows * columns);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
}

template<class T>
void MultiplicationMatricesGPU::setArgs(T *matrixA, T *matrixB, int rows,
                                        int columns, int generalSize) {
    createBuffersAndQueue<T>(rows, columns, generalSize);
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(T) * rows * generalSize, &matrixA[0]);
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(T) * generalSize * columns, &matrixB[0]);
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferResult);
    kernel.setArg(3, sizeof(int), &rows);
    kernel.setArg(4, sizeof(int), &columns);
    kernel.setArg(5, sizeof(int), &generalSize);

}

template<class T>
void MultiplicationMatricesGPU::getResult(T *matrixResult, int rows, int columns) {
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, sizeof(float) * rows * columns, &matrixResult[0]);
}


#endif //COMPARISON_OF_MULTIPLICATIONS_MULTIPLICATIONMATRICESGPU_H
