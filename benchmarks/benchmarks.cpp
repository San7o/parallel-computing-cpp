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
#include <pc/check_symm.hpp>
#include <tenno/ranges.hpp>
#include <tenno/random.hpp>
#include <valfuzz/valfuzz.hpp>

#include <iostream>
#include <cstdlib>    /* exit */
#include <omp.h>

#define PC_MATRIX_MAX_SIZE 1<<12
#define PC_RANDOM_MATRIX_SIZE 256


/*============================================*\
|                     SETUP                    |
\*============================================*/

typedef float** matrix;

/* Using a global matrix so that it will be
 * initialized only once */
matrix matrix_in;
matrix matrix_out;

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

matrix matrix_alloc(tenno::size N)
{
  matrix M = new float*[N];
  for (unsigned int i = 0; i < N; ++i)
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
  for (unsigned int i = 0; i < N; ++i)
    delete[] M[i];
  delete[]  M;
}

/* Execute this before any benchmark */
BEFORE()
{
  matrix_in = matrix_alloc(PC_MATRIX_MAX_SIZE);
  matrix_init(matrix_in, PC_MATRIX_MAX_SIZE);
  matrix_out = matrix_alloc(PC_MATRIX_MAX_SIZE);

  MPI_Init();
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank != 0)
    {
      std::cerr << "Error initializing: Rank is not 0" << endl;
      std::exit(1);
    }
}
/* Execute this after all benchmarks */
AFTER()
{
  matrix_free(matrix_in, PC_MATRIX_MAX_SIZE);
  matrix_free(matrix_out, PC_MATRIX_MAX_SIZE);

  MPI_Finalize();
}


/*============================================*\
|                   TRANSPOSE                  |
\*============================================*/

BENCHMARK(transpose_benchmark,
	  "matTranspose base")
{
  for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTranspose(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_half_benchmark,
	  "matTranspose half")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeHalf(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_columns_benchmark,
	  "matTranspose columns")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeColumns(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_cyclic_benchmark,
	  "matTranspose cyclic")
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
	  "matTranspose intrinsic cyclic")
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
	  "matTranspose intrinsic")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeIntrinsic(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_vectorization_benchmark,
	  "matTranspose vectorization")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeVectorization(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_unrolling_outer_benchmark,
	  "matTranspose unrolling outer")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp2(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_cyclic_unrolled_benchmark,
	  "matTranspose cyclic unrolled")
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
		    pc::matTransposeCyclicUnrolled(arr_in, arr_out, (1<<N)));
    }

    delete[] arr_in;
    delete[] arr_out;
}


BENCHMARK(transpose_omp_4_benchmark,
	  "matTranspose omp 4")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp4(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_8_benchmark,
	  "matTranspose omp 8")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp8(matrix_in, matrix_out, (1<<N)));
    }
}
BENCHMARK(transpose_omp_16_benchmark,
	  "matTranspose omp 16")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp16(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_32_benchmark,
	  "matTranspose omp 32")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp32(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_64_benchmark,
	  "matTranspose omp 64")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp64(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_2_collapse_benchmark,
	  "matTranspose omp 2 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp2Collapse(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_4_collapse_benchmark,
	  "matTranspose omp 4 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp4Collapse(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_8_collapse_benchmark,
	  "matTranspose omp 8 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp8Collapse(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_16_collapse_benchmark,
	  "matTranspose omp 16 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp16Collapse(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_32_collapse_benchmark,
	  "matTranspose omp 32 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp32Collapse(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_64_collapse_benchmark,
	  "matTranspose omp 64 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp64Collapse(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_16_sched_static_benchmark,
	  "matTranspose omp 16 sched static")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp16SchedStatic(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_16_sched_dynamic_benchmark,
	  "matTranspose omp 16 sched dynamic")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp16SchedDynamic(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_omp_16_sched_guided_benchmark,
	  "matTranspose omp 16 sched guided")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::matTransposeOmp16SchedGuided(matrix_in, matrix_out, (1<<N)));
    }
}


/*============================================*\
|                   CHECK SYMM                 |
\*============================================*/

BENCHMARK(check_sym_benchmark, "checkSymm base")
{
    // Make the matrix symmetric (worst case scenario)
    // from now on the input matrix will be symmetric
    // and there is no need to change it again
    for (auto i : tenno::range(PC_MATRIX_MAX_SIZE))
    {
        for (auto j : tenno::range(i, PC_MATRIX_MAX_SIZE))
        {
            matrix_in[i][j] = matrix_out[j][i];
        }
    }

    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSym(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_columns_benchmark,
	  "checkSymm columns")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymColumns(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_vectorization_benchmark,
	  "checkSymm vectorization")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymVectorization(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_unrolling_outer_benchmark,
	  "checkSymm unrolling outer")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymUnrollingOuter(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_unrolling_inner_benchmark,
	  "checkSymm unrolling inner")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymUnrollingInner(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_unrolling_outer_branchless_benchmark,
	  "checkSymm unrolling outer branchless")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymUnrollingOuterNoBranch(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_unrolling_inner__branchless_benchmark,
	  "checkSymm unrolling inner branchless")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymUnrollingInnerNoBranch(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_2_benchmark,
	  "checkSymm omp 2")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp2(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_4_benchmark,
	  "checkSymm omp 4")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp4(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_8_benchmark,
	  "checkSymm omp 8")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp8(matrix_in, (1<<N)));
    }
}
BENCHMARK(check_sym_omp_16_benchmark,
	  "checkSymm omp 16")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp16(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_32_benchmark,
	  "checkSymm omp 32")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp32(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_64_benchmark,
	  "checkSymm omp 64")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp64(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_2_collapse_benchmark,
	  "checkSymm omp 2 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp2Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_4_collapse_benchmark,
	  "checkSymm omp 4 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp4Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_8_collapse_benchmark,
	  "checkSymm omp 8 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp8Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_16_collapse_benchmark,
	  "checkSymm omp 16 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp16Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_32_collapse_benchmark,
	  "checkSymm omp 32 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp32Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_64_collapse_benchmark,
	  "checkSymm omp 64 collapse")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp64Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_16_sched_static_benchmark,
	  "checkSymm omp 16 sched static")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp16SchedStatic(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_16_sched_dynamic_benchmark,
	  "checkSymm omp 16 sched dynamic")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp16SchedDynamic(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_16_sched_guided_benchmark,
	  "checkSymm omp 16 sched guided")
{
    for (size_t N = 2; N <= 12; ++N)
    {
      RUN_BENCHMARK((1<<N),
		    pc::checkSymOmp16SchedGuided(matrix_in, (1<<N)));
    }
}
