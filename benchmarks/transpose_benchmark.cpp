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

#include <pc/transpose.hpp>
#include <valfuzz/valfuzz.hpp>

BENCHMARK(transpose_benchmark, "matrix transpose base")
{
    std::size_t N = 100;
    std::size_t M = 100;
    int **matrix = new int *[N];
    int **out = new int *[M];
    for (std::size_t i = 0; i < N; i++)
    {
        matrix[i] = new int[N];
        for (std::size_t j = 0; j < N; j++)
        {
            matrix[i][j] = valfuzz::get_random<int>();
        }
    }
    for (std::size_t i = 0; i < M; i++)
    {
        out[i] = new int[N];
    }

    RUN_BENCHMARK(2*2*sizeof(int), pc::matrix_transpose(matrix, out, 2, 2));
    RUN_BENCHMARK(4*4*sizeof(int), pc::matrix_transpose(matrix, out, 4, 4));
    RUN_BENCHMARK(8*8*sizeof(int), pc::matrix_transpose(matrix, out, 8, 8));
    RUN_BENCHMARK(16*16*sizeof(int), pc::matrix_transpose(matrix, out, 16, 16));
    RUN_BENCHMARK(32*32*sizeof(int), pc::matrix_transpose(matrix, out, 32, 32));
    RUN_BENCHMARK(32*32*sizeof(int), pc::matrix_transpose(matrix, out, 32, 32));
    RUN_BENCHMARK(64*64*sizeof(int), pc::matrix_transpose(matrix, out, 64, 64));
    RUN_BENCHMARK(80*80*sizeof(int), pc::matrix_transpose(matrix, out, 80, 80));
    RUN_BENCHMARK(100*100*sizeof(int), pc::matrix_transpose(matrix, out, 100, 100));

    delete[] matrix;
    delete[] out;
}
