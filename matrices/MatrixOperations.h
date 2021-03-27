//
// Created by Денис Марценюк on 19.03.2021.
//

#ifndef COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H
#define COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cmath>

class MatrixOperations {

public:
    MatrixOperations();

    template<class T>
            static void generateMatrix(T *matrix, int rows, int columns);
    template<class T>
            static bool compareMatrices(T *matrixA, T *matrixB, int rows, int columns);
    template<class T>
            static void printMatrix(T *matrix, int rows, int columns);
    template<class T>
            static void multiplicationMatrices(T *matrixA, T *matrixB, T *matrixResult, int rows, int columns,
                                       int generalSize);

};

template<class T>
void MatrixOperations::generateMatrix(T *matrix, int rows, int columns) {
    srand(time(0));
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            matrix[i * columns + j] = rand() / (T)RAND_MAX;
        }
    }
}

template<class T>
bool MatrixOperations::compareMatrices(T *matrixA, T *matrixB, int rows, int columns) {
    int counter = 0;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            if (abs(matrixA[i * columns + j] - matrixB[i * columns + j]) <= 0.000001) {
                counter ++;
            }
        }
    }
    return counter == rows * columns;
}

template<class T>
void MatrixOperations::multiplicationMatrices(T *matrixA, T *matrixB, T *matrixResult, int rows, int columns,
                                              int generalSize) {
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

template<class T>
void MatrixOperations::printMatrix(T *matrix, int rows, int columns) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            std::cout << matrix[i * columns + j] << " ";
        }
        std::cout << std::endl;
    }
}


#endif //COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H
