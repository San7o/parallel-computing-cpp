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
#include <mpi.h>
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

/* The MPI world rank and world_size */
int pc::world_rank;
int pc::world_size;

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
	M[i][j] = arr1[i % PC_RANDOM_MATRIX_SIZE] +
	  arr2[j % PC_RANDOM_MATRIX_SIZE];
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

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &pc::world_rank);
  if (pc::world_rank != 0)
    {
      std::cerr << "Error initializing: Rank is not 0" << std::endl;
      MPI_Finalize();
      std::exit(1);
    }
  MPI_Comm_size(MPI_COMM_WORLD, &pc::world_size);
}

/* Execute this after all benchmarks */
AFTER()
{
  matrix_free(pc::matrix_in, PC_MATRIX_MAX_SIZE);
  matrix_free(pc::matrix_out, PC_MATRIX_MAX_SIZE);

  /* Stop the worker */
  char fin[10] = "";
  MPI_Bcast(&fin, 10, MPI_CHAR, 0, MPI_COMM_WORLD);

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
    float* M_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];
    float* T_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];

    constexpr auto arr1 = random_arr1();

    for (size_t i = 0; i < PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE; ++i)
      M_cyclic[i] = arr1[i % PC_RANDOM_MATRIX_SIZE];
      
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeIntrinsicCyclic(M_cyclic, T_cyclic, (1<<N)));
    }
    delete[] M_cyclic;
    delete[] T_cyclic;
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

BENCHMARK(transpose_mpi_benchmark,
	  "matTransposeMPI")
{
    if (pc::world_rank != 0)
      return;

    float *M_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];
    float *T_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];

    constexpr auto arr1 = random_arr1();

    for (size_t i = 0; i < PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE; ++i)
      M_cyclic[i] = arr1[i % PC_RANDOM_MATRIX_SIZE];
      
    int err;
    char message[10] = "Base\0";
    long unsigned int num_iterations =
	valfuzz::get_num_iterations_benchmark() + 2;
    long unsigned int size;
    for (size_t N = 2; N <= 12; ++N)
    {
      err = MPI_Bcast(&message, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;
 
      size = (1<<N);
      err = MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;

      err = MPI_Bcast(&num_iterations, 1,
		       MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;
      
      RUN_BENCHMARK((1<<N),
      	    pc::matTransposeMPI(M_cyclic, T_cyclic, (1<<N)));
    }

    delete[] M_cyclic;
    delete[] T_cyclic;
    return;
}

BENCHMARK(transpose_mpi_nonblocking_benchmark,
	  "matTransposeMPINonblocking")
{
    if (pc::world_rank != 0)
      return;

    float *M_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];
    float *T_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];

    constexpr auto arr1 = random_arr1();

    for (size_t i = 0; i < PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE; ++i)
      M_cyclic[i] = arr1[i % PC_RANDOM_MATRIX_SIZE];
      
    int err;
    char message[10] = "NB\0";
    long unsigned int num_iterations =
	valfuzz::get_num_iterations_benchmark() + 2;
    long unsigned int size;
    for (size_t N = 2; N <= 12; ++N)
    {
      err = MPI_Bcast(&message, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;
 
      size = (1<<N);
      err = MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;

      err = MPI_Bcast(&num_iterations, 1,
		       MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;
      
      RUN_BENCHMARK((1<<N),
      	    pc::matTransposeMPINonblocking(M_cyclic, T_cyclic, (1<<N)));
    }

    delete[] M_cyclic;
    delete[] T_cyclic;
    return;
}

BENCHMARK(transpose_mpi_block_benchmark,
	  "matTransposeMPIBlock")
{
    if (pc::world_rank != 0)
      return;

    float *M_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];
    float *T_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];

    constexpr auto arr1 = random_arr1();

    for (size_t i = 0; i < PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE; ++i)
      M_cyclic[i] = arr1[i % PC_RANDOM_MATRIX_SIZE];
      
    int err;
    char message[10] = "Block\0";
    long unsigned int num_iterations =
	valfuzz::get_num_iterations_benchmark() + 2;
    long unsigned int size;
    for (size_t N = 4; N <= 12; ++N)
    {
      /* Message the workers */
      err = MPI_Bcast(&message, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
      return;

      size = (1<<N);
      err = MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;

      err = MPI_Bcast(&num_iterations, 1,
		       MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;
      
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeMPIBlock(M_cyclic, T_cyclic, (1<<N)));
    }

    delete[] M_cyclic;
    delete[] T_cyclic;
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


BENCHMARK(check_sum_MPI_benchmark,
	  "checkSymMPI")
{
    if (pc::world_rank != 0)
      return;

    float *M_cyclic = new float[PC_MATRIX_MAX_SIZE*PC_MATRIX_MAX_SIZE];

    constexpr auto arr1 = random_arr1();

    for (size_t i = 0; i < PC_MATRIX_MAX_SIZE; ++i)
      for (size_t j = 0; j < PC_MATRIX_MAX_SIZE; ++j)
	{
          M_cyclic[(i*PC_MATRIX_MAX_SIZE) + j] = arr1[i % PC_RANDOM_MATRIX_SIZE];
          M_cyclic[(j*PC_MATRIX_MAX_SIZE) + i] = arr1[i % PC_RANDOM_MATRIX_SIZE];
	}
      
    int err;
    char message[10] = "Sym\0";
    long unsigned int num_iterations =
	valfuzz::get_num_iterations_benchmark() + 2;
    long unsigned int size;
    for (size_t N = 4; N <= 12; ++N)
    {
      /* Message the workers */
      err = MPI_Bcast(&message, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
      return;

      size = (1<<N);
      err = MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;

      err = MPI_Bcast(&num_iterations, 1,
		       MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
        return;
      
      RUN_BENCHMARK((1<<N),
		    pc::checkSymMPI(M_cyclic, (1<<N)));
    }

    delete[] M_cyclic;
    return;
}
