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
 * furnished to do so, subject to the following conditions: *
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
#include <pc/benchmarks.hpp>
#include <pc/check_symm.hpp>
#include <pc/mpi.hpp>
#include <tenno/ranges.hpp>
#include <tenno/random.hpp>
#include <valfuzz/valfuzz.hpp>

#include <iostream>
#include <cstdlib>    /* exit */

#define PC_MATRIX_MAX_SIZE 1<<12
#define PC_RANDOM_MATRIX_SIZE 256


/*============================================*\
|                     SETUP                    |
\*============================================*/

/* Using a global matrix so that it will be
 * initialized only once */
pc::matrix pc::matrix_in;
pc::matrix pc::matrix_out;

/* The MPI world rank */
int pc::world_rank;

/**
 * Generating two random arrays at compile time for
 * faster initialization
 */
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

pc::matrix matrix_alloc(tenno::size N)
{
  pc::matrix M = new float*[N];
  for (unsigned int i = 0; i < N; ++i)
    {
      M[i] = new float[N];
    }
  return M;
}

void matrix_init(pc::matrix M, tenno::size N)
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

void matrix_free(pc::matrix M, tenno::size N)
{
  for (unsigned int i = 0; i < N; ++i)
    delete[] M[i];
  delete[]  M;
}

/* Execute this before any benchmark */
BEFORE()
{
  pc::matrix_in = matrix_alloc(PC_MATRIX_MAX_SIZE);
  matrix_init(pc::matrix_in, PC_MATRIX_MAX_SIZE);
  pc::matrix_out = matrix_alloc(PC_MATRIX_MAX_SIZE);

  mpi::Init(NULL, NULL);
  mpi::Comm_rank(MPI_COMM_WORLD, &pc::world_rank);
  if (pc::world_rank != 0)
    {
      std::cerr << "Error initializing: Rank is not 0" << std::endl;
      MPI_Finalize();
      std::exit(1);
    }
}

/* Execute this after all benchmarks */
AFTER()
{
  matrix_free(pc::matrix_in, PC_MATRIX_MAX_SIZE);
  matrix_free(pc::matrix_out, PC_MATRIX_MAX_SIZE);

  MPI_Finalize();
}


/*============================================*\
|                   TRANSPOSE                  |
\*============================================*/

// Sequencial

BENCHMARK(transpose_benchmark,
	  "matTranspose")
{
  for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTranspose(pc::matrix_in, pc::matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_half_benchmark,
	  "matTransposeHalf")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeHalf(pc::matrix_in, pc::matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_columns_benchmark,
	  "matTransposeColumns")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeColumns(pc::matrix_in, pc::matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_cyclic_benchmark,
	  "matTransposeCyclic")
{
    /* Initialize the vector */
    float* arr_in = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];
    float* arr_out = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];

    constexpr auto arr1 = random_arr1();

    for (size_t i = 0; i < PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE; ++i)
      arr_in[i] = arr1[i % PC_RANDOM_MATRIX_SIZE];
      
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeCyclic(arr_in, arr_out, (1<<N)));
    }

    delete[] arr_in;
    delete[] arr_out;
}

BENCHMARK(transpose_4x4_intrinsic_cyclic_benchmark,
	  "matTransposeIntrinsicCyclic")
{
    /* Initialize the vector */
    float* arr_in = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];
    float* arr_out = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];

    constexpr auto arr1 = random_arr1();

    for (size_t i = 0; i < PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE; ++i)
      arr_in[i] = arr1[i % PC_RANDOM_MATRIX_SIZE];
      
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeIntrinsicCyclic(arr_in, arr_out, (1<<N)));
    }

    delete[] arr_in;
    delete[] arr_out;
}

BENCHMARK(transpose_4x4_intrinsic_benchmark,
	  "matTransposeIntrinsic")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeIntrinsic(pc::matrix_in, pc::matrix_out, (1<<N)));
    }
}

// MPI

BENCHMARK(transpose_mpi,
	  "matTransposeMPIInvert2")
{
    if (pc::world_rank != 0)
      return;

    /* Message the workers */
    char message[10] = "Invert2\0";
    int err = mpi::Bcast(&message, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (err != mpi::SUCCESS)
      return;

    for (size_t N = 2; N <= 12; ++N)
    {
      err = mpi::Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (err != mpi::SUCCESS)
        return;

      RUN_BENCHMARK((1<<N),
		    pc::matTransposeMPIInvert2(pc::matrix_in, pc::matrix_out, (1<<N)));

    }

    int fin = -1;
    err = mpi::Bcast(&fin, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (err != mpi::SUCCESS)
      return;
}


/*============================================*\
|                   CHECK SYMM                 |
\*============================================*/

BENCHMARK(check_sym_benchmark, "checkSymm")
{
    // Make the matrix symmetric (worst case scenario)
    // from now on the input matrix will be symmetric
    // and there is no need to change it again
    for (auto i : tenno::range(PC_MATRIX_MAX_SIZE))
    {
        for (auto j : tenno::range(i, PC_MATRIX_MAX_SIZE))
        {
	  pc::matrix_in[i][j] = pc::matrix_out[j][i];
        }
    }

    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSym(pc::matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_columns_benchmark,
	  "checkSymmColumns")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymColumns(pc::matrix_in, (1<<N)));
    }
}
