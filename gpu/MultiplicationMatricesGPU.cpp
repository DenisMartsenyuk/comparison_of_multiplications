//
// Created by Денис Марценюк on 19.03.2021.
//

#include <iostream>
#include "MultiplicationMatricesGPU.h"

int MultiplicationMatricesGPU::init(int deviceNumber) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cout << "Install OpenCl, please" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.size() == 0) {
        std::cout << "Devices not found." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (deviceNumber >= devices.size() || deviceNumber < 0) {
        std::cout << "Device number is incorrect." << std::endl;
        exit(EXIT_FAILURE);
    }

    device = devices[deviceNumber];
    std::cout << "Information about device:" << std::endl;
    std::cout << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Device max compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "Device local mem size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
    std::cout << "Device max work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl << std::endl;

    context = cl::Context(device);
}

std::string MultiplicationMatricesGPU::readKernel(std::string path) {
    std::ifstream inputFile(path);
    if (!inputFile.is_open()) {
        std::cout << "Error opening file:" << path << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string (std::istreambuf_iterator<char>(inputFile), std::istreambuf_iterator<char>());
}

void MultiplicationMatricesGPU::setProgram(std::string path) {
    std::string kernelSource = readKernel(path);
    program = cl::Program(context, kernelSource, true);
    if (program.build({ device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void MultiplicationMatricesGPU::setWorkGroupAndWorkItems(int rowsWorkGroup, int columnsWorkGroup, int rowsMatrix,
                                                         int columnsMatrix) {
    int rowsWorkItems = ceil(rowsMatrix * 1.0 / rowsWorkGroup) * rowsWorkGroup;
    int columnsWorkItems = ceil(columnsMatrix * 1.0 / columnsWorkGroup) * columnsWorkGroup;
    workItems = cl::NDRange(rowsWorkItems, columnsWorkItems);
    workGroup = cl::NDRange(rowsWorkGroup, columnsWorkGroup);
}

void MultiplicationMatricesGPU::setKernel(std::string nameKernel) {
    kernel = cl::Kernel(program, nameKernel.c_str());
}

void MultiplicationMatricesGPU::createBuffersAndQueue(int rows, int columns, int generalSize) {
    bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * rows * generalSize);
    bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * generalSize * columns);
    bufferResult = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * rows * columns);
    queue = cl::CommandQueue(context, device);
}

void MultiplicationMatricesGPU::setArgs(float *matrixA, float *matrixB, int rows,
                                        int columns, int generalSize) {
    createBuffersAndQueue(rows, columns, generalSize);
    cl_int errcode;
    errcode = queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * rows * generalSize, &matrixA[0]);
    errcode |= queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * generalSize * columns, &matrixB[0]);
    errcode |= kernel.setArg(0, bufferA);
    errcode |= kernel.setArg(1, bufferB);
    errcode |= kernel.setArg(2, bufferResult);
    errcode |= kernel.setArg(3, sizeof(int), &rows);
    errcode |= kernel.setArg(4, sizeof(int), &columns);
    errcode |= kernel.setArg(5, sizeof(int), &generalSize);

}

void MultiplicationMatricesGPU::executeKernel() {
    cl_int errcode;
    errcode |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, workItems,  workGroup); //todo убрать ошибки
    errcode |= queue.finish();
}

void MultiplicationMatricesGPU::getResult(float *matrixResult, int rows, int columns) {
    cl_int errcode;
    errcode |= queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, sizeof(float) * rows * columns, &matrixResult[0]);
}