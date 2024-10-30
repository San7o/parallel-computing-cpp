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

typedef float **matrix;

float **init_matrix(tenno::size N)
{
    float **M = new float *[N];
    for (auto i : tenno::range(N))
    {
        M[i] = new float[N];
        for (auto j : tenno::range(N))
        {
            M[i][j] = valfuzz::get_random<float>();
        }
    }
    return M;
}

void delete_matrix(matrix M, tenno::size N)
{
    for (auto i : tenno::range(N))
    {
        delete[] M[i];
    }
    delete[] M;
}

BENCHMARK(transpose_benchmark, "matrix transpose base")
{
    tenno::size N = 32;
    matrix M = init_matrix(N);
    matrix T = init_matrix(N);

    RUN_BENCHMARK(N * N * sizeof(float), pc::matTranspose(M, T, N));

    delete_matrix(M, N);
    delete_matrix(T, N);
}

BENCHMARK(check_sym_benchmark, "check symmetry base")
{
    tenno::size N = 32;
    matrix M = init_matrix(N);

    // Make the matrix symmetric (worst case scenario)
    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(i, N))
        {
            M[i][j] = M[j][i];
        }
    }

    RUN_BENCHMARK(N * N * sizeof(float), pc::checkSym(M, N));

    delete_matrix(M, N);
}

// #pragma omp somedirective [clause [[,] clause ] ...] new-line
// 
// examples
// pragma omp for
// - private(list)
// - reduction([reduction-modifier,] reduction-identifier: list)
// - schedule(type[, chunk])

#ifdef _OPENMP
#include <omp.h>
#endif
/*
// Check openMP number of threads
BENCHMARK(print_threads_benchmark, "print threads")
{
#ifdef _OPENMP
  int i = omp_get_thread_num();
#else
  int i = -1;
#endif
  std::cout << "Threads: " << i << "\n";
}
*/

// Vector sum
void vector_sum(float *vector, std::size_t size, float *sum)
{
  for (std::size_t i = 0; i < size; i++)
    {
      *sum += vector[i];
    }
}

void vector_sum_parallel(float *vector, std::size_t size, float *sum)
{
  #pragma omp for
  for (std::size_t i = 0; i < size; i++)
    {
      // #pragma omp critical
      // With critical, the performance is incredibly slower
      // because the threads need to synchronyze
      *sum += vector[i];
    }
}

BENCHMARK(vecotr_sum, "vector sum")
{
  const std::size_t size = 100000;
  float *vector = new float[size];
  for (std::size_t i = 0; i < size; i++)
    {
      //is clear???
      vector[i] = (float) i / 2; 
    }
  float sum = 0;
  RUN_BENCHMARK(size*sizeof(float), vector_sum(vector, size, &sum));
  RUN_BENCHMARK(size*sizeof(float), vector_sum_parallel(vector, size, &sum));
  delete[] vector;
}
