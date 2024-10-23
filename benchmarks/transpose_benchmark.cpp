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
    float **M = new float *[N];
    float **T = new float *[N];
    for (std::size_t i = 0; i < N; i++)
    {
        M[i] = new float[N];
        for (std::size_t j = 0; j < N; j++)
        {
            M[i][j] = valfuzz::get_random<float>();
        }
    }
    for (std::size_t i = 0; i < N; i++)
    {
        T[i] = new float[N];
    }

    RUN_BENCHMARK(2 * 2 * sizeof(float), pc::matTranspose(M, T, 2));
    RUN_BENCHMARK(4 * 4 * sizeof(float), pc::matTranspose(M, T, 4));
    RUN_BENCHMARK(8 * 8 * sizeof(float), pc::matTranspose(M, T, 8));
    RUN_BENCHMARK(16 * 16 * sizeof(float), pc::matTranspose(M, T, 16));
    RUN_BENCHMARK(32 * 32 * sizeof(float), pc::matTranspose(M, T, 32));
    RUN_BENCHMARK(64 * 64 * sizeof(float), pc::matTranspose(M, T, 64));
    RUN_BENCHMARK(80 * 80 * sizeof(float), pc::matTranspose(M, T, 80));
    RUN_BENCHMARK(100 * 100 * sizeof(float), pc::matTranspose(M, T, 100));

    delete[] M;
    delete[] T;
}

BENCHMARK(check_sym_benchmark, "check symmetry base")
{
    std::size_t N = 100;
    float **M = new float *[N];
    for (std::size_t i = 0; i < N; i++)
    {
        M[i] = new float[N];
        for (std::size_t j = 0; j < N; j++)
        {
            M[i][j] = valfuzz::get_random<float>();
        }
    }
    // Make the matrix symmetric (worst case scenario)
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = i; j < N; j++)
        {
            M[i][j] = M[j][i];
        }
    }

    RUN_BENCHMARK(2 * 2 * sizeof(float), pc::checkSym(M, 2));
    RUN_BENCHMARK(4 * 4 * sizeof(float), pc::checkSym(M, 4));
    RUN_BENCHMARK(8 * 8 * sizeof(float), pc::checkSym(M, 8));
    RUN_BENCHMARK(16 * 16 * sizeof(float), pc::checkSym(M, 16));
    RUN_BENCHMARK(32 * 32 * sizeof(float), pc::checkSym(M, 32));
    RUN_BENCHMARK(64 * 64 * sizeof(float), pc::checkSym(M, 64));
    RUN_BENCHMARK(80 * 80 * sizeof(float), pc::checkSym(M, 80));
    RUN_BENCHMARK(100 * 100 * sizeof(float), pc::checkSym(M, 100));

    delete[] M;
}
