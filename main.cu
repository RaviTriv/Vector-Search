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

__global__ void findVectorDif(int *matrix, int rows, int cols, int blocks, int *minDistance, int *searchVector)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = threadIdx.x; j < cols; j += cols)
        {
            matrix[i * cols + j] = (int)pow(matrix[i * cols + j] - searchVector[j], 2);
        }
    }
}

__global__ void sumDif(int *matrix, int *tempMatrix, int rows, int cols, int blocks)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        for (int j = 0; j < cols; j++)
        {
            tempMatrix[i] = tempMatrix[i] + matrix[i * cols + j];
        }
    }
}

__global__ void findClosestVector(int rows, int blocks, int *minDistance, int *idx, int *tempMatrix)
{
    for (int i = blockIdx.x; i < rows; i += blocks)
    {
        if ((int)sqrt(tempMatrix[i]) < *minDistance)
        {
            *minDistance = (int)sqrt(tempMatrix[i]);
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
    int *minDistance;
    int *idx;
    int rows = 1 << 20 << 5;
    int cols = 50;

    cudaMalloc(&matrix, rows * cols * sizeof(int));
    cudaMalloc(&searchVector, cols * sizeof(int));
    cudaMallocManaged(&minDistance, sizeof(int));
    cudaMalloc(&tempMatrix, rows * sizeof(int));

    *minDistance = 2147483647;
    printf("GPU\n");
    printf("ROWS: %d\n", rows);
    printf("COLUMNS: %d\n", cols);

    int blocks = 10000;
    setupMatrix<<<blocks, cols>>>(matrix, searchVector, tempMatrix, rows, cols, blocks);
    cudaDeviceSynchronize();
    findVectorDif<<<blocks, cols>>>(matrix, rows, cols, blocks, minDistance, searchVector);
    cudaDeviceSynchronize();
    sumDif<<<blocks, 1>>>(matrix, tempMatrix, rows, cols, blocks);
    cudaDeviceSynchronize();

    cudaFree(matrix);
    cudaMallocManaged(&idx, sizeof(int));
    findClosestVector<<<blocks, 1>>>(rows, blocks, minDistance, idx, tempMatrix);
    cudaDeviceSynchronize();

    cudaFree(searchVector);
    cudaFree(tempMatrix);
    cudaFree(minDistance);

    printf("NEAREST VECTOR INDEX: %d\n", *idx);
    cudaFree(idx);
    timedif = (((double)clock()) / CLOCKS_PER_SEC) - time1;
    printf("TIME TAKEN: %f\n", timedif);
}