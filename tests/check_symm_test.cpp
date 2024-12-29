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

#include <pc/check_symm.hpp>
#include <tenno/ranges.hpp>
#include <valfuzz/valfuzz.hpp>
#include <mpi.h>
#include <pc/benchmarks.hpp>  /* contains definition of matrices and world_rank */

TEST(check_sym_test, "checkSym")
{
    tenno::size N = 10;
    float **M = new float *[N];
    for (auto i : tenno::range(N))
    {
        M[i] = new float[N];
        for (auto j : tenno::range(N))
        {
            M[i][j] = valfuzz::get_random<float>();
        }
    }

    ASSERT(pc::checkSym(M, N) == false);

    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(N))
        {
            M[i][j] = M[j][i];
        }
    }

    ASSERT(pc::checkSym(M, N) == true);
}

TEST(check_sym_columns_test, "checkSymColumns")
{
    tenno::size N = 10;
    float **M = new float *[N];
    for (auto i : tenno::range(N))
    {
        M[i] = new float[N];
        for (auto j : tenno::range(N))
        {
            M[i][j] = valfuzz::get_random<float>();
        }
    }

    ASSERT(pc::checkSymColumns(M, N) == false);

    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(N))
        {
            M[i][j] = M[j][i];
        }
    }

    ASSERT(pc::checkSymColumns(M, N) == true);
}

TEST(check_sym_mpi_test, "checkSymMPI")
{
    if (pc::world_rank != 0)
      ASSERT(false);

    constexpr tenno::size N = (1<<6);
    /* stack allocated matrix */
    float *M_cyclic = new float[N*N];
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
      {
	M_cyclic[i*N + j] = float(i);
	M_cyclic[j*N + i] = float(i);
      }

    /*
    fprintf(stdout, "Original: \n");
    for (size_t i = 0; i < N*N; ++i)
      {
	if (i % N == 0 && i != 0)
	  fprintf(stdout, "\n");
	fprintf(stdout, "%f ", M_cyclic[i]);
      }
    fprintf(stdout, "\n");
    */

    /* Message the workers */
    char message[10] = "Sym\0";
    int err = MPI_Bcast(&message, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
      return;

    size_t n = N; /* -fpermissive gets angry */
    err = MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
      return;

    long unsigned int num_iterations = 1;
    err = MPI_Bcast(&num_iterations, 1,
                     MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
      return;

    bool res = pc::checkSymMPI(M_cyclic, N);

    ASSERT(res);

    delete[] M_cyclic;
    return;
}
