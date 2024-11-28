#include <iostream>
#include <math.h>
#include <time.h>
#include <stdio.h>

double time1, timedif;

__global__ void setupMatrix(int *matrix, int *searchVector, int *sqrtMatrix, int rows, int cols, int blocks)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = threadIdx.x; j < cols; j += cols)
        {
            if (i == 21)
            {
                matrix[i * cols + j] = 4;
                searchVector[j] = 2;
            }
            else
            {
                matrix[i * cols + j] = j + 2;
            }
        }
        sqrtMatrix[i] = 0;
    }
}

__global__ void searchMatrix(int *matrix, int rows, int cols, int blocks, double *min, int *idx, int *searchVector)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = threadIdx.x; j < cols; j += cols)
        {
            matrix[i * cols + j] = (int)pow(matrix[i * cols + j] - searchVector[j], 2);
        }
    }
}

__global__ void sMatrix(int *matrix, int *sqrtMatrix, int rows, int cols, int blocks)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = 0; j < cols; j++)
        {
            sqrtMatrix[i] = sqrtMatrix[i] + matrix[i * cols + j];
        }
    }
}

__global__ void closestVector(int rows, int blocks, double *min, int *idx, int *sqrtMatrix)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        if (sqrt(sqrtMatrix[i]) < *min)
        {
            *min = sqrt(sqrtMatrix[i]);
            *idx = i;
        };
    }
}

int main()
{
    time1 = (double)clock();
    time1 = time1 / CLOCKS_PER_SEC;
    int *matrix;
    int *sqrtMatrix;
    int *searchVector;
    double *min;
    int *idx;
    int rows = 1 << 20 << 5;
    int cols = 50;

    cudaMalloc(&matrix, rows * cols * sizeof(int));
    cudaMallocManaged(&searchVector, cols * sizeof(int));
    cudaMallocManaged(&min, sizeof(double));
    cudaMallocManaged(&idx, sizeof(int));
    cudaMalloc(&sqrtMatrix, rows * sizeof(int));

    *min = 21.0;
    printf("GPU\n");
    printf("ROWS: %d\n", rows);
    printf("COLUMNS: %d\n", cols);

    int blocks = 10000;
    setupMatrix<<<blocks, cols>>>(matrix, searchVector, sqrtMatrix, rows, cols, blocks);
    cudaDeviceSynchronize();

    searchMatrix<<<blocks, cols>>>(matrix, rows, cols, blocks, min, idx, searchVector);
    cudaDeviceSynchronize();

    sMatrix<<<blocks, 1>>>(matrix, sqrtMatrix, rows, cols, blocks);
    cudaDeviceSynchronize();

    closestVector<<<blocks, 1>>>(rows, blocks, min, idx, sqrtMatrix);
    cudaDeviceSynchronize();

    cudaFree(searchVector);
    cudaFree(sqrtMatrix);
    cudaFree(min);

    printf("NEAREST VECTOR INDEX: %d\n", *idx);
    cudaFree(idx);
    timedif = (((double)clock()) / CLOCKS_PER_SEC) - time1;
    printf("TIME TAKEN: %f\n", timedif);
}