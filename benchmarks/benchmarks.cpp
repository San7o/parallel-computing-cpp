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
/* Execute this after all benchmarks */
AFTER()
{
  matrix_free(matrix_in, PC_MATRIX_MAX_SIZE);
  matrix_free(matrix_out, PC_MATRIX_MAX_SIZE);
}

/*============================================*\
|                   TRANSPOSE                  |
\*============================================*/

BENCHMARK(transpose_benchmark, "matrix transpose base")
{
  for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::matTranspose(matrix_in, matrix_out, (1<<N)));
    }
}

BENCHMARK(transpose_half_benchmark, "matrix transpose half")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::matTransposeHalf(matrix_in, matrix_out, (1<<N)));
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

    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSym(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_vectorization_benchmark,
	  "checkSymm vectorization")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymVectorization(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_unrolling_outer_benchmark,
	  "checkSymm unrolling outer")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymUnrollingOuter(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_unrolling_inner_benchmark,
	  "checkSymm unrolling inner")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymUnrollingInner(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_unrolling_outer_no_branchless_benchmark,
	  "checkSymm unrolling outer branchless")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymUnrollingOuterNoBranch(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_unrolling_inner_no_branchless_benchmark,
	  "checkSymm unrolling inner branchless")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymUnrollingInnerNoBranch(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_2_benchmark,
	  "checkSymm omp 2")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp2(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_4_benchmark,
	  "checkSymm omp 4")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp4(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_8_benchmark,
	  "checkSymm omp 8")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp8(matrix_in, (1<<N)));
    }
}
BENCHMARK(check_sym_omp_16_benchmark,
	  "checkSymm omp 16")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp16(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_32_benchmark,
	  "checkSymm omp 32")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp32(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_64_benchmark,
	  "checkSymm omp 64")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp64(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_2_collapse_benchmark,
	  "checkSymm omp 2 collapse")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp2Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_4_collapse_benchmark,
	  "checkSymm omp 4 collapse")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp4Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_8_collapse_benchmark,
	  "checkSymm omp 8 collapse")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp8Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_16_collapse_benchmark,
	  "checkSymm omp 16 collapse")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp16Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_32_collapse_benchmark,
	  "checkSymm omp 32 collapse")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp32Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_64_collapse_benchmark,
	  "checkSymm omp 64 collapse")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp64Collapse(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_16_sched_static_benchmark,
	  "checkSymm omp 16 sched static")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp16SchedStatic(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_16_sched_dynamic_benchmark,
	  "checkSymm omp 16 sched dynamic")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp16SchedDynamic(matrix_in, (1<<N)));
    }
}

BENCHMARK(check_sym_omp_16_sched_guided_benchmark,
	  "checkSymm omp 16 sched guided")
{
    for (size_t N = 2; N < 12; ++N)
    {
      RUN_BENCHMARK((1<<N) * (1<<N) * sizeof(float),
		    pc::checkSymOmp16SchedGuided(matrix_in, (1<<N)));
    }
}
