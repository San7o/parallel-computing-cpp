#include <pc/check_symm.hpp>
#include <cstdint>
#include <omp.h>

/*============================================*\
|                     NOTES                    |
\*============================================*/
/*
 * This file fontains the algorithms to compute
 * if an N*N matrix is symmetric. There are
 * 3 types of algorithms in this file:
 * - baseline: basic algorithms without any
 *       optimizaitons
 * - implicit optimizations: vectorization and
 *       prefetching
 * - explicit optimizations: OMP
 */ 

/*============================================*\
|                   BASELINE                   |
\*============================================*/

bool pc::checkSym(float **M, tenno::size N)
{
  bool symm = true;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      if (M[i][j] != M[j][i])
	    symm = false;
  return symm;
}

bool pc::checkSymColumns(float **M, tenno::size N)
{
  bool symm = true;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      if (M[j][i] != M[i][j])
	    symm = false;
  return symm;
}

// TODO: MPI
