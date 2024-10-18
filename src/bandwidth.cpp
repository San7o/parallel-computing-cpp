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


void pc::bubble_sort(int *a, int *out, int size)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = a[i];
    }
    for (int i = 0; i < size - 1; i++)
    {
        for (int j = 0; j < size - i - 1; j++)
        {
            if (out[j] > out[j + 1])
            {
                int temp = out[j];
                out[j] = out[j + 1];
                out[j + 1] = temp;
            }
        }
    }
}

void merge(int *a, int left, int middle, int right)
{
    int i, j, k;
    int n1 = middle - left + 1;
    int n2 = right - middle;

    int *L = new int[n1];
    int *R = new int[n2];

    for (i = 0; i < n1; i++)
    {
        L[i] = a[left + i];
    }
    for (j = 0; j < n2; j++)
    {
        R[j] = a[middle + 1 + j];
    }

    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            a[k] = L[i];
            i++;
        }
        else
        {
            a[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        a[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        a[k] = R[j];
        j++;
        k++;
    }

    delete [] L;
    delete [] R;
    return;
}

void merge_sort_recursive(int *a, int left, int right)
{
    if (left < right)
    {
        int middle = left + (right - left) / 2;
        merge_sort_recursive(a, left, middle);
        merge_sort_recursive(a, middle + 1, right);
        merge(a, left, middle, right);
    }
    return;
}

void pc::merge_sort(int *a, int *out, int size)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = a[i];
    }
    merge_sort_recursive(out, 0, size - 1);
    return;
}

