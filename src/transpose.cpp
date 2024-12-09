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

#include <pc/transpose.hpp>
#include <tenno/ranges.hpp>
#include <omp.h>
#include <immintrin.h>         /* For AVX intrinsics */


/*============================================*\
|                   BASELINE                   |
\*============================================*/

void pc::matTranspose(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}

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

void pc::matTransposeCyclic(float *M, float *T, tenno::size N) {
    for(long unsigned int n = 0; n<N*N; n++) {
        long unsigned int i = n/N;
        long unsigned int j = n%N;
        T[n] = M[N*j + i];
    }
}

/*
Linux strikes again arch/x86/crypto/aria-aesni-avc2-asm_64.S
#define transpose_4x4(x0, x1, x2, x3, t1, t2)		\
	vpunpckhdq x1, x0, t2;				\
	vpunpckldq x1, x0, x0;				\
							\
	vpunpckldq x3, x2, t1;				\
	vpunpckhdq x3, x2, x2;				\
							\
	vpunpckhqdq t1, x0, x1;				\
	vpunpcklqdq t1, x0, x0;				\
							\
	vpunpckhqdq x2, t2, x3;				\
	vpunpcklqdq x2, t2, x2;

transpose_4x4_f32_intrinsic is the intrinsic version of the above code
*/
void transpose_4x4_f32_intrinsic(const float *src0,
                                 const float *src1, 
                                 const float *src2,
                                 const float *src3,
                                 float *dst0,
                                 float *dst1,
                                 float *dst2,
                                 float *dst3) {

    __m128 row0 = _mm_loadu_ps(src0); // Load Rows
    __m128 row1 = _mm_loadu_ps(src1);
    __m128 row2 = _mm_loadu_ps(src2);
    __m128 row3 = _mm_loadu_ps(src3);

    // Unpack and interleave rows
    __m128 t0 = _mm_unpacklo_ps(row0, row1);
    __m128 t1 = _mm_unpackhi_ps(row0, row1);
    __m128 t2 = _mm_unpacklo_ps(row2, row3);
    __m128 t3 = _mm_unpackhi_ps(row2, row3);

    // Shuffle to form transposed rows
    __m128 d0 = _mm_movelh_ps(t0, t2);
    __m128 d1 = _mm_movehl_ps(t2, t0);
    __m128 d2 = _mm_movelh_ps(t1, t3);
    __m128 d3 = _mm_movehl_ps(t3, t1);

    // Store transposed rows
    _mm_storeu_ps(dst0, d0);
    _mm_storeu_ps(dst1, d1);
    _mm_storeu_ps(dst2, d2);
    _mm_storeu_ps(dst3, d3);
}

void pc::matTransposeIntrinsicCyclic(float *mat_in, float *mat_out, size_t N)
{
    for (size_t i = 0; i < N; i += 4) {
        for (size_t j = 0; j < N; j += 4) {
            transpose_4x4_f32_intrinsic(
                            &mat_in[i * N + j], 
                            &mat_in[(i + 1) * N + j], 
                            &mat_in[(i + 2) * N + j], 
                            &mat_in[(i + 3) * N + j], 
                            &mat_out[j * N + i],
                            &mat_out[(j + 1) * N + i],
                            &mat_out[(j + 2) * N + i],
                            &mat_out[(j + 3) * N + i]);
        }
    }
}

void pc::matTransposeIntrinsic(float **mat_in, float **mat_out, size_t N)
{
    for (size_t i = 0; i < N; i += 4) {
        for (size_t j = 0; j < N; j += 4) {
            transpose_4x4_f32_intrinsic(
                            &mat_in[i][j], 
                            &mat_in[i + 1][j], 
                            &mat_in[i + 2][j], 
                            &mat_in[i + 3][j], 
                            &mat_out[j][i],
                            &mat_out[j + 1][i],
                            &mat_out[j + 2][i],
                            &mat_out[j + 3][i]);
        }
    }
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
      for (tenno::size j = 0; j < N; j += 4)
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
  for (tenno::size i = 0; i < N; i += 4)
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
      for (tenno::size j = i; j < N; j += 4)
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
  for (tenno::size i = 0; i < N; i +=4)
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

void pc::matTransposeCyclicUnrolled(float *M, float *T, tenno::size N) {
  for(long unsigned int n = 0; n<N*N - 4; n+=4) {
        T[n] = M[N*(n%N) + (n/N)];
        T[n+1] = M[N*((n+1)%N) + ((n+1)/N)];
        T[n+2] = M[N*((n+2)%N) + ((n+2)/N)];
        T[n+3] = M[N*((n+3)%N) + ((n+3)/N)];
    }
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


/*============================================*\
|                     MPI                      |
\*============================================*/


void matTransposeMPI(float **M, float **T, tenno::size N)
{
  char message[10] = "Ciaone\0";
  int err = MPI_Bcast(&message, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS)
    return;
  
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[i][j] = M[j][i];
  return;
}
