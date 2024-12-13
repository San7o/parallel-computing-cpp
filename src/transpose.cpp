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
#include <algorithm>


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

void pc::matTransposeColumns(float **M, float **T, tenno::size N)
{
  for (tenno::size i = 0; i < N; ++i)
      for (tenno::size j = 0; j < N; ++j)
            T[j][i] = M[i][j];
  return;
}

void pc::matTransposeCyclic(float *M, float *T, tenno::size N) {
    for(long unsigned int n = 0; n<N*N; ++n) {
        T[n] = M[N*(n%N) + n/N];
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

void pc::matTransposeMPI(float *M, float *T, tenno::size N)
{
  MPI_Datatype row_t;
  int err = mpi::Type_contiguous((int) N,   /* count   */
				 MPI_FLOAT, /* oldtype */
				 &row_t);   /* newtype */
  if (err != mpi::SUCCESS)
    return;
  err = mpi::Type_commit(&row_t);
  if (err != mpi::SUCCESS)
    return;

  MPI_Datatype col_t_tmp, col_t;
  err = mpi::Type_vector((int) N,        /* count       */
			 1,              /* blocklength */
			 (int) N,        /* stride      */
			 MPI_FLOAT,      /* oldtype     */
			 &col_t_tmp);    /* newtype     */
  if (err != mpi::SUCCESS)
    return;
  err = mpi::Type_create_resized(col_t_tmp,     /* oldtype */
				 0,             /* lb      */
				 sizeof(float), /* extent  */
				 &col_t);       /* newtype */
  err = mpi::Type_commit(&col_t);
  if (err != mpi::SUCCESS)
    return;
  mpi::Type_free(&col_t_tmp);

  float *row = new float[N * N / world_size];
  err = mpi::Scatter(M,                      /* sendbuf   */
		     (int) (N / world_size), /* sendcount */
		     row_t,                  /* sendtype  */
		     row,                    /* recvbuf   */
		     (int) (N / world_size), /* recvcount */
		     row_t,                  /* recvtype  */
		     0,                      /* root      */
 		     MPI_COMM_WORLD);        /* comm      */
  if (err != mpi::SUCCESS)
    return;

  err = mpi::Gather(row,                  /* sendbuf   */
		    (int) N / world_size, /* sendcount */
		    row_t,                /* sendtype  */
		    T,                    /* recvbuf   */
		    (int) N / world_size, /* recvcount */
		    col_t,                /* recvtype  */
		    0,                    /* root      */
		    MPI_COMM_WORLD);      /* comm      */
  if (err != mpi::SUCCESS)
    return;

  delete[] row;
  mpi::Type_free(&row_t);
  mpi::Type_free(&col_t);
  return;
}

void pc::matTransposeMPIBlock(float *M, float *T, tenno::size N, tenno::size B)
{
  (void) B;
  MPI_Datatype row_t;
  int err = mpi::Type_contiguous((int) N,   /* count   */
				 MPI_FLOAT, /* oldtype */
				 &row_t);   /* newtype */
  if (err != mpi::SUCCESS)
    return;
  err = mpi::Type_commit(&row_t);
  if (err != mpi::SUCCESS)
    return;

  MPI_Datatype col_t_tmp, col_t;
  err = mpi::Type_vector((int) N,        /* count       */
			 1,              /* blocklength */
			 (int) N,        /* stride      */
			 MPI_FLOAT,      /* oldtype     */
			 &col_t_tmp);    /* newtype     */
  if (err != mpi::SUCCESS)
    return;
  err = mpi::Type_create_resized(col_t_tmp,     /* oldtype */
				 0,             /* lb      */
				 sizeof(float), /* extent  */
				 &col_t);       /* newtype */
  err = mpi::Type_commit(&col_t);
  if (err != mpi::SUCCESS)
    return;
  mpi::Type_free(&col_t_tmp);

  float *row = new float[N * N / world_size];
  err = mpi::Scatter(M,                      /* sendbuf   */
		     (int) (N / world_size), /* sendcount */
		     row_t,                  /* sendtype  */
		     row,                    /* recvbuf   */
		     (int) (N / world_size), /* recvcount */
		     row_t,                  /* recvtype  */
		     0,                      /* root      */
 		     MPI_COMM_WORLD);        /* comm      */
  if (err != mpi::SUCCESS)
    return;

  err = mpi::Gather(row,                  /* sendbuf   */
		    (int) N / world_size, /* sendcount */
		    row_t,                /* sendtype  */
		    T,                    /* recvbuf   */
		    (int) N / world_size, /* recvcount */
		    col_t,                /* recvtype  */
		    0,                    /* root      */
		    MPI_COMM_WORLD);      /* comm      */
  if (err != mpi::SUCCESS)
    return;

  delete[] row;
  mpi::Type_free(&row_t);
  mpi::Type_free(&col_t);
  return;
}
