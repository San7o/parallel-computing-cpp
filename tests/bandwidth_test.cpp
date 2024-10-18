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

TEST(bandwidth_matrix_multiply_test, "Matrix Multiply")
{
    const int N = 2;

    float matrix1[4] = {1, 2, 3, 4};
    float matrix2[4] = {5, 6, 7, 8};
    float result[4];

    float expected[4] = {19, 22, 43, 50};

    pc::matrix_multiply(matrix1, matrix2, result, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ASSERT(result[i * N + j] == expected[i * N + j]);
        }
    }

}

TEST(bandwidth_bubble_sort_test, "Bubble Sort")
{
    const int N = 5;

    int array[5] = {64, 34, 25, 12, 22};
    int out[5];
    int expected[5] = {12, 22, 25, 34, 64};

    pc::bubble_sort(array, out, N);

    for (int i = 0; i < N; i++) {
        ASSERT(out[i] == expected[i]);
    }
}

TEST(bandwidth_merge_sort_test, "Merge Sort")
{
    const int N = 5;

    int array[5] = {64, 34, 25, 12, 22};
    int out[5];
    int expected[5] = {12, 22, 25, 34, 64};

    pc::merge_sort(array, out, N);

    for (int i = 0; i < N; i++) {
        ASSERT(out[i] == expected[i]);
    }
}
