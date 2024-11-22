/*============================================*\
|                     NOTES                    |
\*============================================*/
/*
 * This file fontains the algorithms to compute
 * the transposition of a N*N matrix. There are
 * 3 types of algorithms in this file:
 * - baseline: basic algorithms without any
 *       optimizaitons
 * - implicit optimizations: vectorization and
 *       prefetching
 * - explicit optimizations: OMP
 */ 

/*
void transpose(float *src, float *dst, const int N, const int M) {
    #pragma omp parallel for
    for(int n = 0; n<N*M; n++) {
        int i = n/N;
        int j = n%N;
        dst[n] = src[M*j + i];
    }
}

// With blocks
inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
}
*/

#include <pc/transpose.hpp>
#include <tenno/ranges.hpp>
#include <omp.h>

/*============================================*\
|                   BASELINE                   |
\*============================================*/

/*
 * matTranspose
 *
 * For each element of the matrix with column i and row j,
 * the algorithm copies the value in the output matrix
 * in column j and row i.
 * Time Complexity: O(n^2)
 * Space Complexity: O(n^2)
 * Assumption: square matrix
 */
void pc::matTranspose(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

/*
 * matTransposeHalf
 *
 * An alternative algorithm to matTranspose is to iterate
 * over only the upper half of the matrix with respect
 * to the diagonal. This cuts in half the number of iterations
 * but the number of memory accesses (both read and writes)
 * do not change, as well as the time and space complexity.
 * Despite this, this algorithm improves from the previous
 * one by about 33%
 * Time Complexity: O(n^2)
 * Space Complexity: O(n^2)
 * Assumption: square matrix
 */
void pc::matTransposeHalf(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = i; j < N; ++j)
        {
            T[i][j] = M[j][i];
            T[j][i] = M[i][j];
        }
  return;
}

/* access pattern */

void pc::matTransposeColumns(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[j][i] = M[i][j];
  return;

}


/*============================================*\
|              IMPLICIT PARALLELISM            |
\*============================================*/

void pc::matTransposeVectorization(float **M, float **T, tenno::size N)
{
  #pragma omp simd
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeUnrolledInner(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; j = j + 4)
	{
            T[i][j] = M[j][i];
            T[i][j+1] = M[j+1][i];
            T[i][j+2] = M[j+2][i];
            T[i][j+3] = M[j+3][i];
	}
  return;
}

void pc::matTransposeUnrolledOuter(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; i = i + 4)
      for (tenno::size j = 0; j < N; ++j)
	{
            T[i][j] = M[j][i];
            T[i+1][j] = M[j][i+1];
            T[i+2][j] = M[j][i+2];
            T[i+3][j] = M[j][i+3];
	}
  return;
}

/* half */

void pc::matTransposeHalfVectorization(float **M, float **T, tenno::size N)
{
  #pragma omp simd
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = i; j < N; ++j)
        {
            T[i][j] = M[j][i];
            T[j][i] = M[i][j];
        }
  return;
}

void pc::matTransposeHalfUnrolledInner(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = i; j < N; j = j + 4)
        {
            T[i][j] = M[j][i];
            T[i][j+1] = M[j+1][i];
            T[i][j+2] = M[j+2][i];
            T[i][j+3] = M[j+3][i];

            T[j][i] = M[i][j];
            T[j+1][i] = M[i][j+1];
            T[j+2][i] = M[i][j+2];
            T[j+3][i] = M[i][j+3];
        }
  return;
}

void pc::matTransposeHalfUnrolledOuter(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; i = i + 4)
      for (tenno::size j = i; j < N; ++j)
        {
            T[i][j] = M[j][i];
            T[i+1][j] = M[j][i+1];
            T[i+2][j] = M[j][i+2];
            T[i+3][j] = M[j][i+3];

            T[j][i] = M[i][j];
            T[j][i+1] = M[i+1][j];
            T[j][i+2] = M[i+2][j];
            T[j][i+3] = M[i+3][j];
        }
  return;
}


/*============================================*\
|              EXPLICIT PARALLELISM            |
\*============================================*/


void pc::matTransposeOmp2(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(2);
  #pragma omp parallel for
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp4(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(4);
  #pragma omp parallel for
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp8(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(8);
  #pragma omp parallel for
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp16(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(16);
  #pragma omp parallel for
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp32(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(32);
  #pragma omp parallel for
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp64(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(64);
  #pragma omp parallel for
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp2Collapse(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(2);
#pragma omp parallel for collapse(2)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp4Collapse(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(4);
#pragma omp parallel for collapse(2)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp8Collapse(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(8);
#pragma omp parallel for collapse(2)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp16Collapse(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(16);
#pragma omp parallel for collapse(2)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp32Collapse(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(32);
#pragma omp parallel for collapse(2)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp64Collapse(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(64);
#pragma omp parallel for collapse(2)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp16SchedStatic(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(16);
#pragma omp parallel for schedule(static, 16)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp16SchedDynamic(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(16);
#pragma omp parallel for schedule(dynamic, 16)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

void pc::matTransposeOmp16SchedGuided(float **M, float **T, tenno::size N)
{
  omp_set_num_threads(16);
#pragma omp parallel for schedule(guided, 16)
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

