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
#include <tenno/ranges.hpp>
#include <valfuzz/valfuzz.hpp>

TEST(transpose_matrix_test, "Matrix Transpose")
{
    tenno::size N = 10;
    float **M = new float *[N];
    float **T = new float *[N];
    for (auto i : tenno::range(N))
    {
        M[i] = new float[N];
        for (auto j : tenno::range(N))
        {
            M[i][j] = valfuzz::get_random<float>();
        }
    }
    for (auto i : tenno::range(N))
    {
        T[i] = new float[N];
    }

    pc::matTranspose(M, T, N);

    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(N))
        {
            ASSERT(M[i][j] == T[j][i]);
        }
    }
}

TEST(transpose_matrix_half_test, "Matrix Transpose Half")
{
    tenno::size N = 10;
    float **M = new float *[N];
    float **T = new float *[N];
    for (auto i : tenno::range(N))
    {
        M[i] = new float[N];
        for (auto j : tenno::range(N))
        {
            M[i][j] = valfuzz::get_random<float>();
        }
    }
    for (auto i : tenno::range(N))
    {
        T[i] = new float[N];
    }

    pc::matTransposeHalf(M, T, N);

    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(N))
        {
            ASSERT(M[i][j] == T[j][i]);
        }
    }
}
