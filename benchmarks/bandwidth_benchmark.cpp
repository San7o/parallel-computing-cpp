/*
 * MIT License
 *
 * Copyright (c) 2024 Giovanni Santini
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include <pc/bandwidth.hpp>
#include <valfuzz/valfuzz.hpp>

BENCHMARK(bandwidth_benchmark, "write and read")
{
    int size = 100000;
    int *arr = new int[size];

    RUN_BENCHMARK(100*sizeof(int), pc::write_and_read(arr, 100));
    RUN_BENCHMARK(1000*sizeof(int), pc::write_and_read(arr, 1000));
    RUN_BENCHMARK(10000*sizeof(int), pc::write_and_read(arr, 10000));
    RUN_BENCHMARK(50000*sizeof(int), pc::write_and_read(arr, 50000));
    RUN_BENCHMARK(100000*sizeof(int), pc::write_and_read(arr, 100000));

    delete[] arr;
}

BENCHMARK(bandwidth_write_serial, "write serial")
{
    int size = 100000;
    int *arr = new int[size];

    RUN_BENCHMARK(100, pc::write_serial(arr, 100));
    RUN_BENCHMARK(1000, pc::write_serial(arr, 1000));
    RUN_BENCHMARK(10000, pc::write_serial(arr, 10000));
    RUN_BENCHMARK(50000, pc::write_serial(arr, 50000));
    RUN_BENCHMARK(100000, pc::write_serial(arr, 100000));

    delete[] arr;
}

BENCHMARK(bandwidth_write_random, "write random")
{
    int size = 100000;
    int *arr = new int[size];
    int *indexes = new int[size];

    for (int i = 0; i < size; i++)
    {
        indexes[i] = valfuzz::get_random<int>() % size;
    }

    std::cout << "Random indexes generated" << std::endl;

    RUN_BENCHMARK(100, pc::write_random(arr, indexes, 100));
    RUN_BENCHMARK(1000, pc::write_random(arr, indexes, 1000));
    RUN_BENCHMARK(10000, pc::write_random(arr, indexes, 10000));
    RUN_BENCHMARK(50000, pc::write_random(arr, indexes, 50000));
    RUN_BENCHMARK(100000, pc::write_random(arr, indexes, 100000));

    delete[] arr;
    delete[] indexes;
}

BENCHMARK(bandwidth_flops_benchmark, "float sum")
{
    int size = 100000;
    float *a = new float[size];
    float *b = new float[size];
    float *c = new float[size];

    RUN_BENCHMARK(100, pc::float_sum(a, b, c, 100));
    RUN_BENCHMARK(1000, pc::float_sum(a, b, c, 1000));
    RUN_BENCHMARK(10000, pc::float_sum(a, b, c, 10000));
    RUN_BENCHMARK(50000, pc::float_sum(a, b, c, 50000));
    RUN_BENCHMARK(100000, pc::float_sum(a, b, c, 100000));

    delete[] a;
    delete[] b;
    delete[] c;
}

BENCHMARK(bandwidth_matrix_multiplication, "matrix multiplication")
{
    int size = 100;
    float *matrix1 = new float[size * size];
    float *matrix2 = new float[size * size];
    float *result = new float[size * size];

    for (int i = 0; i < size * size; i++)
    {
        matrix1[i] = valfuzz::get_random<float>();
        matrix2[i] = valfuzz::get_random<float>();
    }

    RUN_BENCHMARK(10, pc::matrix_multiply(matrix1, matrix2, result, 10));
    RUN_BENCHMARK(25, pc::matrix_multiply(matrix1, matrix2, result, 25));
    RUN_BENCHMARK(50, pc::matrix_multiply(matrix1, matrix2, result, 50));
    RUN_BENCHMARK(75, pc::matrix_multiply(matrix1, matrix2, result, 75));
    RUN_BENCHMARK(100, pc::matrix_multiply(matrix1, matrix2, result, 100));
}

BENCHMARK(bandwidth_bubble_sort, "bubble sort")
{
    int size = 100000;
    int *arr = new int[size];
    int *out = new int[size];

    for (int i = 0; i < size; i++)
    {
        arr[i] = valfuzz::get_random<int>();
    }

    RUN_BENCHMARK(100, pc::bubble_sort(arr, out, 100));
    RUN_BENCHMARK(1000, pc::bubble_sort(arr, out, 1000));
    RUN_BENCHMARK(10000, pc::bubble_sort(arr, out, 10000));
    RUN_BENCHMARK(50000, pc::bubble_sort(arr, out, 50000));
    RUN_BENCHMARK(100000, pc::bubble_sort(arr, out, 100000));

    delete[] arr;
    delete[] out;
}

BENCHMARK(bandwidth_merge_sort, "merge sort")
{
    int size = 100000;
    int *arr = new int[size];
    int *out = new int[size];

    for (int i = 0; i < size; i++)
    {
        arr[i] = valfuzz::get_random<int>();
    }

    RUN_BENCHMARK(100, pc::merge_sort(arr, out, 100));
    RUN_BENCHMARK(1000, pc::merge_sort(arr, out, 1000));
    RUN_BENCHMARK(10000, pc::merge_sort(arr, out, 10000));
    RUN_BENCHMARK(50000, pc::merge_sort(arr, out, 50000));
    RUN_BENCHMARK(100000, pc::merge_sort(arr, out, 100000));

    delete[] arr;
    delete[] out;
}
