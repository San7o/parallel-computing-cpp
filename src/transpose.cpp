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
#include <mpi.h>
#include <tenno/ranges.hpp>
#include <immintrin.h>         /* For AVX intrinsics */
#include <algorithm>
#include <math.h>
#include <chrono>
#include <iostream>
#include <fstream>


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
  if (world_size > (int) N) /* fallback */
  {
    if (world_rank == 0)
    {
      for (tenno::size i = 0; i < N*N; ++i)
	T[i] = M[N*(i % N) + (i / N)];
    }
    return;
  }

  MPI_Datatype row_t;
  int err = MPI_Type_contiguous((int) N,   /* count   */
				 MPI_FLOAT, /* oldtype */
				 &row_t);   /* newtype */
  if (err != MPI_SUCCESS)
    return;
  err = MPI_Type_commit(&row_t);
  if (err != MPI_SUCCESS)
    return;

  MPI_Datatype col_t_tmp, col_t;
  err = MPI_Type_vector((int) N,        /* count       */
			 1,              /* blocklength */
			 (int) N,        /* stride      */
			 MPI_FLOAT,      /* oldtype     */
			 &col_t_tmp);    /* newtype     */
  if (err != MPI_SUCCESS)
    return;
  err = MPI_Type_create_resized(col_t_tmp,     /* oldtype */
				 0,             /* lb      */
				 sizeof(float), /* extent  */
				 &col_t);       /* newtype */
  err = MPI_Type_commit(&col_t);
  if (err != MPI_SUCCESS)
    return;
  MPI_Type_free(&col_t_tmp);

  float *row = new float[N * N / world_size];
  err = MPI_Scatter(M,                      /* sendbuf   */
		     (int) (N / world_size), /* sendcount */
		     row_t,                  /* sendtype  */
		     row,                    /* recvbuf   */
		     (int) (N / world_size), /* recvcount */
		     row_t,                  /* recvtype  */
		     0,                      /* root      */
 		     MPI_COMM_WORLD);        /* comm      */
  if (err != MPI_SUCCESS)
    return;

  err = MPI_Gather(row,                  /* sendbuf   */
		    (int) N / world_size, /* sendcount */
		    row_t,                /* sendtype  */
		    T,                    /* recvbuf   */
		    (int) N / world_size, /* recvcount */
		    col_t,                /* recvtype  */
		    0,                    /* root      */
		    MPI_COMM_WORLD);      /* comm      */
  if (err != MPI_SUCCESS)
    return;

  delete[] row;
  MPI_Type_free(&row_t);
  MPI_Type_free(&col_t);
  return;
}

void pc::matTransposeMPINonblocking(float *M, float *T, tenno::size N)
{
  if (world_size > (int) N) /* fallback */
  {
    if (world_rank == 0)
    {
      for (tenno::size i = 0; i < N*N; ++i)
	T[i] = M[N*(i % N) + (i / N)];
    }
    return;
  }

  MPI_Datatype row_t;
  int err = MPI_Type_contiguous((int) N,   /* count   */
				 MPI_FLOAT, /* oldtype */
				 &row_t);   /* newtype */
  if (err != MPI_SUCCESS)
    return;
  err = MPI_Type_commit(&row_t);
  if (err != MPI_SUCCESS)
    return;

  MPI_Datatype col_t_tmp, col_t;
  err = MPI_Type_vector((int) N,        /* count       */
			 1,              /* blocklength */
			 (int) N,        /* stride      */
			 MPI_FLOAT,      /* oldtype     */
			 &col_t_tmp);    /* newtype     */
  if (err != MPI_SUCCESS)
    return;
  err = MPI_Type_create_resized(col_t_tmp,     /* oldtype */
				 0,             /* lb      */
				 sizeof(float), /* extent  */
				 &col_t);       /* newtype */
  err = MPI_Type_commit(&col_t);
  if (err != MPI_SUCCESS)
    return;
  MPI_Type_free(&col_t_tmp);

  MPI_Request request;
  float *row = new float[N * N / world_size];
  err = MPI_Iscatter(M,                      /* sendbuf   */
		     (int) (N / world_size), /* sendcount */
		     row_t,                  /* sendtype  */
		     row,                    /* recvbuf   */
		     (int) (N / world_size), /* recvcount */
		     row_t,                  /* recvtype  */
		     0,                      /* root      */
 		     MPI_COMM_WORLD,         /* comm      */
		     &request);              /* request   */
  if (err != MPI_SUCCESS)
    return;
  MPI_Wait(&request, NULL);

  err = MPI_Igather(row,                  /* sendbuf   */
		    (int) N / world_size, /* sendcount */
		    row_t,                /* sendtype  */
		    T,                    /* recvbuf   */
		    (int) N / world_size, /* recvcount */
		    col_t,                /* recvtype  */
		    0,                    /* root      */
		    MPI_COMM_WORLD,       /* comm      */
		    &request);
  if (err != MPI_SUCCESS)
    return;
  MPI_Wait(&request, NULL);

  delete[] row;
  MPI_Type_free(&row_t);
  MPI_Type_free(&col_t);
  return;
}
void pc::matTransposeMPIBlock(float *M, float *T, tenno::size N)
{
  if (world_size > (int) (N * N) || world_size < 4) /* fallback */
  {
    if (world_rank == 0)
    {
      for (tenno::size i = 0; i < N*N; ++i)
	T[i] = M[N*(i % N) + (i / N)];
    }
    return;
  }

  /* Calculate the displacement */
  int block_side = (int)N/int(sqrt(world_size));
  float *block = new float[block_side * block_side];
  int *displacements = new int[world_size];
  int *displacements_transposed = new int[world_size];
  int *counts = new int[world_size];
  for (int i = 0; i < world_size; ++i)
  { /* Displacement of block types, assuming a stride of sizeof(float) beween 2 blocks */
    displacements[i] =
                 ((i*block_side)%(int)N)        /* col */
               + (i*block_side/(int)N)*(int)N*block_side; /* row    */
    displacements_transposed[i] =
                 ((i*block_side)%(int)N)*(int)N /* col */
               + (i*block_side/(int)N)*block_side;       /*  row */
    counts[i] = 1;
  }

  /* Debug */
  /*
  if (world_rank == 0)
  {
  printf("Displacements:\n");
  for (int i = 0; i < world_size; ++i)
      printf("%d ", displacements[i]);
  printf("\n");
  for (int i = 0; i < world_size; ++i)
      printf("%d ", displacements_transposed[i]);
  printf("\n");
  }
  */

  //printf("Lots of processes\n");

  MPI_Datatype block_t_tmp, block_t;
  int err = MPI_Type_vector(block_side, /* count       */
                         block_side,    /* blocklength */
			 (int)N,        /* stride      */
			 MPI_FLOAT,     /* oldtype     */
			 &block_t_tmp); /* newtype     */
  if (err != MPI_SUCCESS)
  {
    delete[] block;
    delete[] counts;
    delete[] displacements;
    delete[] displacements_transposed;
    return;
  }
  err = MPI_Type_create_resized(block_t_tmp,     /* oldtype */
				 0,             /* lb      */
				 sizeof(float), /* extent  */
				 &block_t);       /* newtype */
  err = MPI_Type_commit(&block_t);
  if (err != MPI_SUCCESS)
  {
    delete[] block;
    delete[] counts;
    delete[] displacements;
    delete[] displacements_transposed;
    return;
  }
  MPI_Type_free(&block_t_tmp);

  //printf("Scattering\n");
  
  err = MPI_Scatterv(M,                /* sendbuf       */
		     counts,           /* sendcount     */
                     displacements,    /* displacements */
		     block_t,          /* sendtype      */
		     block,            /* recvbuf       */
		     block_side * block_side, /* recvcount */
		     MPI_FLOAT,        /* recvtype      */
		     0,                /* root          */
 		     MPI_COMM_WORLD);  /* comm          */
  if (err != MPI_SUCCESS)
    goto end;

  //printf("Scattered\n");
  /*
  if (pc::world_rank == 0)
  {
  for (size_t i = 0; i < block_side * block_side; ++i)
  {
      if (i % block_side == 0 && i != 0)
        fprintf(stdout, "\n");
      fprintf(stdout, "%f ", block[i]);
  }
  printf("\n");
  }
  */

  /* Transpose the block */
  //printf("Transposed:\n");
  for (int i = 0; i < (int) (block_side); ++i)
    for (int j = i; j < (int) (block_side); ++j)
        std::swap(block[block_side*i + j], block[j*block_side + i]);
  /*
  if (pc::world_rank == 0)
  {
  fprintf(stdout, "Transposed: \n");
  for (size_t i = 0; i < block_side * block_side; ++i)
  {
      if (i % block_side == 0 && i != 0)
        fprintf(stdout, "\n");
      fprintf(stdout, "%f ", block[i]);
  }
  printf("\n");
  }
  */

  // printf("Gathering\n");

  err = MPI_Gatherv(block,                 /* sendbuf   */
		    block_side * block_side,                    /* sendcount */
		    MPI_FLOAT,              /* sendtype  */
		    T,                    /* recvbuf   */
		    counts,           /* recvcount */
		    displacements_transposed, /* displacements */
		    block_t,   /* recvtype  */
		    0,                    /* root      */
		    MPI_COMM_WORLD);      /* comm      */
  if (err != MPI_SUCCESS)
    goto end;

  //printf("Gathered\n");

end:
  delete[] block;
  delete[] displacements;
  delete[] displacements_transposed;
  delete[] counts;
  MPI_Type_free(&block_t);
  return;
}

void pc::matTransposeMPIBlockDebug(float *M, float *T, tenno::size N)
{
  if (world_size > (int) (N * N) || world_size < 4) /* fallback */
  {
    if (world_rank == 0)
    {
      for (tenno::size i = 0; i < N*N; ++i)
	T[i] = M[N*(i % N) + (i / N)];
    }
    return;
  }

  
  /* Calculate the displacement */

  auto start = std::chrono::high_resolution_clock::now();
  
  int block_side = (int)N/int(sqrt(world_size));
  float *block = new float[block_side * block_side];
  int *displacements = new int[world_size];
  int *displacements_transposed = new int[world_size];
  int *counts = new int[world_size];
  for (int i = 0; i < world_size; ++i)
  { /* Displacement of block types, assuming a stride of sizeof(float) beween 2 blocks */
    displacements[i] =
                 ((i*block_side)%(int)N)        /* col */
               + (i*block_side/(int)N)*(int)N*block_side; /* row    */
    displacements_transposed[i] =
                 ((i*block_side)%(int)N)*(int)N /* col */
               + (i*block_side/(int)N)*block_side;       /*  row */
    counts[i] = 1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> displacement = end - start;

  /* Debug */
  /*
  if (world_rank == 0)
  {
  printf("Displacements:\n");
  for (int i = 0; i < world_size; ++i)
      printf("%d ", displacements[i]);
  printf("\n");
  for (int i = 0; i < world_size; ++i)
      printf("%d ", displacements_transposed[i]);
  printf("\n");
  }
  */

  //printf("Lots of processes\n");

  start = std::chrono::high_resolution_clock::now();

  MPI_Datatype block_t_tmp, block_t;
  int err = MPI_Type_vector(block_side, /* count       */
                         block_side,    /* blocklength */
			 (int)N,        /* stride      */
			 MPI_FLOAT,     /* oldtype     */
			 &block_t_tmp); /* newtype     */
  if (err != MPI_SUCCESS)
  {
    delete[] block;
    delete[] counts;
    delete[] displacements;
    delete[] displacements_transposed;
    return;
  }
  err = MPI_Type_create_resized(block_t_tmp,     /* oldtype */
				 0,             /* lb      */
				 sizeof(float), /* extent  */
				 &block_t);       /* newtype */
  err = MPI_Type_commit(&block_t);
  if (err != MPI_SUCCESS)
  {
    delete[] block;
    delete[] counts;
    delete[] displacements;
    delete[] displacements_transposed;
    return;
  }
  MPI_Type_free(&block_t_tmp);
  
  end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> setup = end - start;
  std::chrono::duration<double> scatter, transpose, gather;

  //printf("Scattering\n");

  start = std::chrono::high_resolution_clock::now();
  
  err = MPI_Scatterv(M,                /* sendbuf       */
		     counts,           /* sendcount     */
                     displacements,    /* displacements */
		     block_t,          /* sendtype      */
		     block,            /* recvbuf       */
		     block_side * block_side, /* recvcount */
		     MPI_FLOAT,        /* recvtype      */
		     0,                /* root          */
 		     MPI_COMM_WORLD);  /* comm          */
  if (err != MPI_SUCCESS)
    goto end;

  end = std::chrono::high_resolution_clock::now();
  scatter = end - start;

  //printf("Scattered\n");
  /*
  if (pc::world_rank == 0)
  {
  for (size_t i = 0; i < block_side * block_side; ++i)
  {
      if (i % block_side == 0 && i != 0)
        fprintf(stdout, "\n");
      fprintf(stdout, "%f ", block[i]);
  }
  printf("\n");
  }
  */

  /* Transpose the block */

  start = std::chrono::high_resolution_clock::now();
  
  //printf("Transpose\n");
  for (int i = 0; i < (int) (block_side); ++i)
    for (int j = i; j < (int) (block_side); ++j)
        std::swap(block[block_side*i + j], block[j*block_side + i]);

  end = std::chrono::high_resolution_clock::now();
  transpose = end - start;
  //printf("Transposed\n");

  /*
  if (pc::world_rank == 0)
  {
  fprintf(stdout, "Transposed: \n");
  for (size_t i = 0; i < block_side * block_side; ++i)
  {
      if (i % block_side == 0 && i != 0)
        fprintf(stdout, "\n");
      fprintf(stdout, "%f ", block[i]);
  }
  printf("\n");
  }
  */

  //printf("Gathering\n");

  start = std::chrono::high_resolution_clock::now();

  err = MPI_Gatherv(block,                 /* sendbuf   */
		    block_side * block_side,                    /* sendcount */
		    MPI_FLOAT,              /* sendtype  */
		    T,                    /* recvbuf   */
		    counts,           /* recvcount */
		    displacements_transposed, /* displacements */
		    block_t,   /* recvtype  */
		    0,                    /* root      */
		    MPI_COMM_WORLD);      /* comm      */
  if (err != MPI_SUCCESS)
    goto end;

  end = std::chrono::high_resolution_clock::now();
  gather = end - start;

  //printf("Gathered\n");

  if (pc::world_rank == 0)
  {
    std::ofstream out("dbg.txt", std::ofstream::out | std::ofstream::app);
    out << displacement.count() << ","
	<< setup.count() << ","
        << scatter.count() << ","
        << transpose.count() << ","
        << gather.count() << std::endl;
    out.close();
  }
  
end:
  delete[] block;
  delete[] displacements;
  delete[] displacements_transposed;
  delete[] counts;
  MPI_Type_free(&block_t);
  return;
}
