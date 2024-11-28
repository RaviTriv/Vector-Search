#include <iostream>
#include <math.h>
#include <time.h>
#include <stdio.h>

double time1, timedif;

__global__ void setupMatrix(int *matrix, int *searchVector, int rows, int M, int blocks)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = threadIdx.x; j < M; j += M)
        {
            if (i == 21)
            {
                matrix[i * M + j] = 4;
                searchVector[j] = 2;
            }
            else
            {
                matrix[i * M + j] = j + 2;
            }
        }
    }
}

__global__ void setupMatrix2(int *sqrtMatrix, int rows, int blocks)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        sqrtMatrix[i] = 0.0;
    }
}

__global__ void searchMatrix(int *matrix, int rows, int M, int blocks, double *min, int *idx, int *searchVector)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = threadIdx.x; j < M; j += M)
        {
            matrix[i * M + j] = (int)pow(matrix[i * M + j] - searchVector[j], 2);
        }
    }
}

__global__ void sMatrix(int *matrix, int *sqrtMatrix, int rows, int M, int blocks)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = 0; j < M; j++)
        {
            sqrtMatrix[i] = sqrtMatrix[i] + matrix[i * M + j];
        }
    }
}

__global__ void closestVector(int rows, int M, int blocks, double *min, int *idx, int *sqrtMatrix)
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
    int M = 50;

    cudaMalloc(&matrix, rows * M * sizeof(int));
    cudaMallocManaged(&searchVector, M * sizeof(int));
    cudaMallocManaged(&min, sizeof(double));
    cudaMallocManaged(&idx, sizeof(int));

    *min = 21.0;
    printf("GPU\n");
    printf("ROWS: %d\n", rows);
    printf("COLUMNS: %d\n", M);


    int blocks = 10000;
    setupMatrix<<<blocks, M>>>(matrix, searchVector, rows, M, blocks);
    cudaDeviceSynchronize();

    searchMatrix<<<blocks, M>>>(matrix, rows, M, blocks, min, idx, searchVector);
    cudaDeviceSynchronize();

    cudaMalloc(&sqrtMatrix, rows * sizeof(int));

    setupMatrix2<<<blocks, 1>>>(sqrtMatrix, rows, blocks);
    cudaDeviceSynchronize();

    sMatrix<<<blocks, 1>>>(matrix, sqrtMatrix, rows, M, blocks);
    cudaDeviceSynchronize();

    closestVector<<<blocks, 1>>>(rows, M, blocks, min, idx, sqrtMatrix);
    cudaDeviceSynchronize();

    printf("NEAREST VECTOR INDEX: %d\n", *idx);
    cudaFree(searchVector);
    cudaFree(sqrtMatrix);
    cudaFree(min);
    cudaFree(idx);
    timedif = (((double)clock()) / CLOCKS_PER_SEC) - time1;
    printf("TIME TAKEN: %f\n", timedif);
}