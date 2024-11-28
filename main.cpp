#include <cstdio>
#include <cmath>
#include <time.h>
#include <stdio.h>

double time1, timedif;

int main()
{
    time1 = (double)clock();
    time1 = time1 / CLOCKS_PER_SEC;
    int distance = 0;
    int rows = 1 << 20 << 5;
    int cols = 50;
    int minDistance = 2147483647;
    int idx = 0;
    int *matrix = new int[rows * cols];
    int *searchVector = new int[cols];

    printf("CPU\n");
    printf("ROWS: %d\n", rows);
    printf("COLUMNS: %d\n", cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (i == 5021)
            {
                *(matrix + i * cols + j) = 4;
                *(searchVector + j) = 2;
            }
            else
            {
                *(matrix + i * cols + j) = j + 2;
            }
        }
    }

    for (int i = 0; i < rows; i++)
    {
        distance = 0;
        for (int j = 0; j < cols; j++)
        {
            distance += pow(*(matrix + i * cols + j) - *(searchVector + j), 2);
        }
        distance = sqrt(distance);
        if (distance < minDistance)
        {
            minDistance = distance;
            idx = i;
        }
    }

    printf("NEAREST VECTOR INDEX: %d\n", idx);
    delete matrix;
    delete searchVector;
    timedif = (((double)clock()) / CLOCKS_PER_SEC) - time1;
    printf("TIME TAKEN: %f\n", timedif);
}
