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
#include <pc/benchmarks.hpp>
#include <pc/mpi.hpp>
#include <tenno/ranges.hpp>
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
|                     MPI                      |
\*============================================*/

void pc::matTransposeMPIInvert2(float **M, float **T, tenno::size N)
{
  (void) T;
  (void) M;

  /* Create a new datatype */
  MPI_Datatype row_t;
  int err = mpi::Type_vector((int) N, (int) N, 0, MPI_FLOAT, &row_t);
  if (err != mpi::SUCCESS)
    return;

  /* Scatter */
  float *row = new float[N*N];
  /*
  err = mpi::Scatter(M, (int) N, row_t, row, 1, row_t, 0, MPI_COMM_WORLD);
  if (err != mpi::SUCCESS)
    return;
  */

  /* Gather */

  /* Create a new datatype */

  /* Scatter */

  /* Gather */

  delete[] row;
  return;
}
