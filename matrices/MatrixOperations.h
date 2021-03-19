//
// Created by Денис Марценюк on 19.03.2021.
//

#ifndef COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H
#define COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H

#include <ctime>
#include <cstdlib>
#include <iostream>

class MatrixOperations {

public:
    MatrixOperations();

    static void generateMatrix(float *matrix, int rows, int columns);
    static bool compareMatrices(float *matrixA, float *matrixB, int rows, int columns);
    static void printMatrix(float *matrix, int rows, int columns);
    static void multiplicationMatrices(float *matrixA, float *matrixB, float *matrixResult, int rows, int columns,
                                       int generalSize);


};


#endif //COMPARISON_OF_MULTIPLICATIONS_MATRIXOPERATIONS_H
