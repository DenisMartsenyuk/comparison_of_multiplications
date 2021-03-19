//
// Created by Денис Марценюк on 19.03.2021.
//

#ifndef COMPARISON_OF_MULTIPLICATIONS_CONFIG_H
#define COMPARISON_OF_MULTIPLICATIONS_CONFIG_H

#define NUMBER_OF_MEASUREMENTS 5

#define ROWS 512
#define COLUMNS 512
#define GENERAL_SIZE 512

#define DEVICE_NUMBER 2
#define PATH_TO_KERNEL_FILE "/Users/mega_user/Desktop/GPU /comparison_of_multiplications/kernels/kernel.cl"
#define WORK_GROUP_ROWS 16
#define WORK_GROUP_COLUMNS 16

#define NAME_KERNEL_1 "simple_multiplication"
#define NAME_KERNEL_2 "optimization_1_multiplication"

#endif //COMPARISON_OF_MULTIPLICATIONS_CONFIG_H
