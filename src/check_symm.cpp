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

/*============================================*\
|              IMPLICIT PARALLELISM            |
\*============================================*/

bool pc::checkSymVectorization(float **M, tenno::size N)
{
  bool symm = true;
  #pragma omp simd /* Already assumes ivdep */
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      if (M[i][j] != M[j][i])
	symm = false;
  return symm;
}

bool pc::checkSymUnrollingOuter(float **M, tenno::size N)
{
  bool symm = true;
  for (size_t i = 0; i < N - 4; i += 4)
    for (size_t j = i; j < N; ++j)
      {
	if (M[i][j] != M[j][i]) symm = false;
	if (M[i+1][j] != M[j][i+1]) symm = false;
	if (M[i+2][j] != M[j][i+2]) symm = false;
	if (M[i+3][j] != M[j][i+3]) symm = false;
      }
  return symm;
}

bool pc::checkSymUnrollingInner(float **M, tenno::size N)
{
  bool symm = true;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N - 4; j += 4)
      {
	if (M[i][j] != M[j][i]) symm = false;
	if (M[i][j+1] != M[j+1][i]) symm = false;
	if (M[i][j+2] != M[j+2][i]) symm = false;
	if (M[i][j+3] != M[j+3][i]) symm = false;
      }
  return symm;
}

bool pc::checkSymUnrollingOuterNoBranch(float **M, tenno::size N)
{
  bool symm = true;
  for (size_t i = 0; i < N - 4; i += 4)
    for (size_t j = i; j < N; ++j)
      {
	symm |=
	        ((
		  ((uint32_t) M[i][j] ^ (uint32_t) M[j][i])
	        | ((uint32_t) M[i+1][j] ^ (uint32_t) M[j][i+1])
                | ((uint32_t) M[i+2][j] ^ (uint32_t) M[j][i+2])
                | ((uint32_t) M[i+3][j] ^ (uint32_t) M[j][i+3]))
         	!= 0); // normalize the results
      }
  return symm;
}

bool pc::checkSymUnrollingInnerNoBranch(float **M, tenno::size N)
{
  bool symm = true;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N - 4; j += 4)
      {
	symm &=
	        ((
		  ((uint32_t) M[i][j] - (uint32_t) M[j][i])
	        | ((uint32_t) M[i][j+1] - (uint32_t) M[j+1][i])
                | ((uint32_t) M[i][j+2] - (uint32_t) M[j+2][i])
                | ((uint32_t) M[i][j+3] - (uint32_t) M[j+3][i]))
         	== 0); // normalize the results
      }
  return symm;
}

/*============================================*\
|              EXPLICIT PARALLELISM            |
\*============================================*/

bool pc::checkSymOmp2(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(2);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }
  return symm;
}

bool pc::checkSymOmp4(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(4);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp8(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(8);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp16(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(16);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }
  
  return symm;
}

bool pc::checkSymOmp32(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(32);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp64(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(64);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp2Collapse(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(2);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for collapse(2)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp4Collapse(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(4);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for collapse(2)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp8Collapse(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(8);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for collapse(2)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp16Collapse(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(16);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for collapse(2)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp32Collapse(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(32);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for collapse(2)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp64Collapse(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(64);

  #pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for collapse(2)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp16SchedStatic(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(16);

#pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for schedule(static, 16)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp16SchedDynamic(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(16);

#pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for schedule(dynamic, 16)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}

bool pc::checkSymOmp16SchedGuided(float **M, tenno::size N)
{
  bool symm = true;
  omp_set_num_threads(16);

#pragma omp parallel shared(symm)
  {
    bool thread_symm = false;
    #pragma omp for schedule(guided, 16)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i; j < N; ++j)
	if (M[i][j] != M[j][i])
	  thread_symm = false;

    #pragma omp critical
    symm &= thread_symm;
  }

  return symm;
}
