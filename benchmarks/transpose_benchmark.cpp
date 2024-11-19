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
#include <tenno/random.hpp>
#include <valfuzz/valfuzz.hpp>

#define PC_MATRIX_MAX_SIZE 2<<12
#define PC_RANDOM_MATRIX_SIZE 256

typedef float** matrix;

/* Using a global matrix so that it will be
 * initialized only once */
matrix matrix_in;
matrix matrix_out;

constexpr tenno::array<float, PC_RANDOM_MATRIX_SIZE> random_arr1()
{
    constexpr unsigned int seed1 = 1337;
    constexpr float min = 0.0f;
    constexpr float max = 0.5f;
    return tenno::random_array<PC_RANDOM_MATRIX_SIZE>(seed1, min, max);
}

constexpr tenno::array<float, PC_RANDOM_MATRIX_SIZE> random_arr2()
{
    constexpr unsigned int seed2 = 892424;
    constexpr float min = 0.0f;
    constexpr float max = 0.5f;
    return tenno::random_array<PC_RANDOM_MATRIX_SIZE>(seed2, min, max);
}

matrix matrix_alloc(tenno::size N)
{
  matrix M = new float*[N];
  for (const auto i : tenno::range(N))
    {
      M[i] = new float[N];
    }
  return M;
}

void matrix_init(matrix M, tenno::size N)
{
  constexpr auto arr1 = random_arr1();
  constexpr auto arr2 = random_arr2();

  for (const auto i : tenno::range(N))
    for (const auto j : tenno::range(N))
      {
	/* Using random, very slow */
	// M[i][j] = valfuzz::get_random<float>();

	/* Using constexpr random, very fast */
	M[i][j] = arr1[i % PC_RANDOM_MATRIX_SIZE] + arr2[j % PC_RANDOM_MATRIX_SIZE];
      }
}

void matrix_free(matrix M, tenno::size N)
{
  for (const auto i : tenno::range(N))
    delete[] M[i];
  delete[]  M;
}

/* Execute this before any benchmark */
BEFORE()
{
  matrix_in = matrix_alloc(PC_MATRIX_MAX_SIZE);
  matrix_init(matrix_in, PC_MATRIX_MAX_SIZE);
  matrix_out = matrix_alloc(PC_MATRIX_MAX_SIZE);
}

AFTER()
{
  matrix_free(matrix_in, PC_MATRIX_MAX_SIZE);
  matrix_free(matrix_out, PC_MATRIX_MAX_SIZE);
}

BENCHMARK(transpose_benchmark, "matrix transpose base")
{
    tenno::size N = 1024;
    RUN_BENCHMARK(N * N * sizeof(float),
		  pc::matTranspose(matrix_in, matrix_out, N));
}

BENCHMARK(transpose_half_benchmark, "matrix transpose half")
{
    tenno::size N = 1024;
    RUN_BENCHMARK(N * N * sizeof(float),
		  pc::matTransposeHalf(matrix_in, matrix_out, N));
}

BENCHMARK(check_sym_benchmark, "check symmetry base")
{
    tenno::size N = 1024;
    // Make the matrix symmetric (worst case scenario)
    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(i, N))
        {
            matrix_in[i][j] = matrix_out[j][i];
        }
    }

    RUN_BENCHMARK(N * N * sizeof(float),
		  pc::checkSym(matrix_in, N));
}
