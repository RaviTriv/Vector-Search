#include <iostream>
#include <math.h>
#include <time.h>
#include <stdio.h>

double time1, timedif;

__global__ void setupMatrix(int *matrix, int *searchVector, int *tempMatrix, int rows, int cols, int blocks)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = threadIdx.x; j < cols; j += cols)
        {
            if (i == 5021)
            {
                matrix[i * cols + j] = 4;
                searchVector[j] = 2;
            }
            else
            {
                matrix[i * cols + j] = j + 2;
            }
        }
        tempMatrix[i] = 0;
    }
}

__global__ void searchMatrix(int *matrix, int rows, int cols, int blocks, int *min, int *idx, int *searchVector)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = threadIdx.x; j < cols; j += cols)
        {
            matrix[i * cols + j] = (int)pow(matrix[i * cols + j] - searchVector[j], 2);
        }
    }
}

__global__ void sMatrix(int *matrix, int *tempMatrix, int rows, int cols, int blocks)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = 0; j < cols; j++)
        {
            tempMatrix[i] = tempMatrix[i] + matrix[i * cols + j];
        }
    }
}

__global__ void closestVector(int rows, int blocks, int *min, int *idx, int *tempMatrix)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        if ((int)sqrt(tempMatrix[i]) < *min)
        {
            *min = (int)sqrt(tempMatrix[i]);
            *idx = i;
        };
    }
}

int main()
{
    time1 = (double)clock();
    time1 = time1 / CLOCKS_PER_SEC;
    int *matrix;
    int *tempMatrix;
    int *searchVector;
    int *min;
    int *idx;
    int rows = 1 << 20 << 5;
    int cols = 50;

    cudaMalloc(&matrix, rows * cols * sizeof(int));
    cudaMallocManaged(&searchVector, cols * sizeof(int));
    cudaMallocManaged(&min, sizeof(double));
    cudaMallocManaged(&idx, sizeof(int));
    cudaMalloc(&tempMatrix, rows * sizeof(int));

    *min = 2147483647;
    printf("GPU\n");
    printf("ROWS: %d\n", rows);
    printf("COLUMNS: %d\n", cols);

    int blocks = 10000;
    setupMatrix<<<blocks, cols>>>(matrix, searchVector, tempMatrix, rows, cols, blocks);
    cudaDeviceSynchronize();

    searchMatrix<<<blocks, cols>>>(matrix, rows, cols, blocks, min, idx, searchVector);
    cudaDeviceSynchronize();

    sMatrix<<<blocks, 1>>>(matrix, tempMatrix, rows, cols, blocks);
    cudaDeviceSynchronize();

    closestVector<<<blocks, 1>>>(rows, blocks, min, idx, tempMatrix);
    cudaDeviceSynchronize();

    cudaFree(searchVector);
    cudaFree(tempMatrix);
    cudaFree(min);

    printf("NEAREST VECTOR INDEX: %d\n", *idx);
    cudaFree(idx);
    timedif = (((double)clock()) / CLOCKS_PER_SEC) - time1;
    printf("TIME TAKEN: %f\n", timedif);
}