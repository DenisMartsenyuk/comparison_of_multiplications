//
// Created by Денис Марценюк on 19.03.2021.
//

#include "MatrixOperations.h"

void MatrixOperations::generateMatrix(float *matrix, int rows, int columns) {
    srand(time(0));
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            matrix[i * columns + j] = rand() / (float)RAND_MAX;
        }
    }
}

void MatrixOperations::multiplicationMatrices(float *matrixA, float *matrixB, float *matrixResult, int rows,
                                              int columns, int generalSize) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            float res = 0.0;
            for(int k = 0; k < generalSize; k++) {
                res += matrixA[i * generalSize + k] * matrixB[k * columns + j];
            }
            matrixResult[i * columns + j] = res;
        }
    }
}

bool MatrixOperations::compareMatrices(float *matrixA, float *matrixB, int rows, int columns) {
    int counter = 0;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            if (matrixA[i * columns + j] == matrixB[i * columns + j]) {
                counter ++;
            }
        }
    }
    return counter == rows * columns;
}

void MatrixOperations::printMatrix(float *matrix, int rows, int columns) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            std::cout << matrix[i * columns + j] << " ";
        }
        std::cout << std::endl;
    }
}