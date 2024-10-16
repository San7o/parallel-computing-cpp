#include <pc/bandwidth.hpp>

void pc::write_and_read(int *a, int size)
{
    for (int i = 0; i < size; i++)
    {
        a[i] = i;
    }
    for (int i = 0; i < size; i++)
    {
        a[i] = a[i] + 1;
    }
    return;
}

void pc::write_serial(int *a, int size)
{
    for (int i = 0; i < size; i++)
    {
        a[i] = i;
    }
    return;
}

void pc::write_random(int *a, int *indexes, int size)
{
    for (int i = 0; i < size; i++)
    {
        a[indexes[i]] = a[indexes[i]] + 1;
    }
    return;
}

void pc::float_sum(float *vector1, float *vector2, float *result, int N)
{
    for (int i = 0; i < N; i++)
    {
        result[i] = vector1[i] + vector2[i];
    }
    return;
}

void pc::matrix_multiply(float *matrix1, float *matrix2, float *result, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i * N + j] = 0;
            for (int k = 0; k < N; k++)
            {
                result[i * N + j] += matrix1[i * N + k] * matrix2[k * N + j];
            }
        }
    }
    return;
}
