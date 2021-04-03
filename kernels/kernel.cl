#define BLOCK_SIZE 16

__kernel void simple_multiplication_float( __global float* matrix_a, __global float* matrix_b, __global float* result, const int rows, const int columns, const int general_size) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row >= rows || col >= columns) {
        return;
    }
    float sum = 0;
    for (int i = 0; i < general_size; i++) {
        sum += matrix_a[row * general_size + i] * matrix_b[i * columns + col];
    }
    result[row * columns + col] = sum;
}

__kernel void optimization_1_multiplication_float(__global float* matrix_a, __global float* matrix_b, __global float* result, const int rows, const int columns, const int general_size) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int x_in_block = get_local_id(0);
    int y_in_block = get_local_id(1);
    __local float sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __local float sub_b[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0;
    for (int i = 0; i < ceil(general_size * 1.0 / BLOCK_SIZE); ++i) {
        int index_a = y * general_size + i * BLOCK_SIZE + x_in_block;
        int index_b = (i * BLOCK_SIZE + y_in_block) * general_size + x;
        if (y < rows && (i * BLOCK_SIZE + x_in_block) < general_size) {
            sub_a[y_in_block][x_in_block] = matrix_a[index_a];
        } else {
            sub_a[y_in_block][x_in_block] = 0;
        }
        if (x < columns && (i * BLOCK_SIZE + y_in_block) < general_size) {
            sub_b[y_in_block][x_in_block] = matrix_b[index_b];
        } else {
            sub_b[y_in_block][x_in_block] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += sub_a[y_in_block][j] * sub_b[j][x_in_block];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (x < rows && y < columns) {
    		result[general_size * y + x] = sum;
    }
}

__kernel void simple_multiplication_double( __global double* matrix_a, __global double* matrix_b, __global double* result, const int rows, const int columns, const int general_size) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row >= rows || col >= columns) {
        return;
    }
    double sum = 0;
    for (int i = 0; i < general_size; i++) {
        sum += matrix_a[row * general_size + i] * matrix_b[i * columns + col];
    }
    result[row * columns + col] = sum;
}

__kernel void optimization_1_multiplication_double(__global double* matrix_a, __global double* matrix_b, __global double* result, const int rows, const int columns, const int general_size) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int x_in_block = get_local_id(0);
    int y_in_block = get_local_id(1);

    __local double sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __local double sub_b[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0;
    for (int i = 0; i < ceil(general_size * 1.0 / BLOCK_SIZE); ++i) {
        int index_a = y * general_size + i * BLOCK_SIZE + x_in_block;
        int index_b = (i * BLOCK_SIZE + y_in_block) * general_size + x;
        sub_a[y_in_block][x_in_block] = matrix_a[index_a];
        sub_b[y_in_block][x_in_block] = matrix_b[index_b];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += sub_a[y_in_block][j] * sub_b[j][x_in_block];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[general_size * y + x] = sum;
}

