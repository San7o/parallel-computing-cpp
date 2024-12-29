#include <pc/check_symm.hpp>
#include <pc/benchmarks.hpp>
#include <mpi.h>
#include <tenno/ranges.hpp>
#include <immintrin.h>         /* For AVX intrinsics */
#include <algorithm>
#include <math.h>


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
|                     MPI                      |
\*============================================*/


bool pc::checkSymMPI(float *M, tenno::size N)
{
  if (world_size > (int) (N * N) || world_size < 4) /* fallback */
  {
    bool symm = true;
    if (world_rank == 0)
    {
      for (size_t i = 0; i < N; ++i)
        for (size_t j = i; j < N; ++j)
          if (M[i*N + j] != M[j*N + i])
	    symm = false;
    }
    return symm;
  }

  /* Calculate the displacement */
  int block_side = (int)N/int(sqrt(world_size));
  float *block = new float[block_side * block_side];
  float *block_transposed = new float[block_side * block_side];
  bool isSymm = true;
  bool res = true;
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
    return false;
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
    return false;
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

  err = MPI_Scatterv(M,                /* sendbuf       */
		     counts,           /* sendcount     */
                     displacements_transposed, /* displacements */
		     block_t,          /* sendtype      */
		     block_transposed, /* recvbuf       */
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

  /* Check the symmetry */
  //printf("Transposed:\n");
  for (int i = 0; i < (int) (block_side); ++i)
    for (int j = 0; j < (int) (block_side); ++j)
      if (block[i*block_side + j] != block_transposed[j * block_side + i])
	isSymm = false;

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

  // printf("Reducing\n");
  err = MPI_Reduce(&isSymm,    /* sendbuf  */
		   &res,       /* recvbuf  */
		   1,          /* count    */
		   MPI_C_BOOL,   /* datatype */
		   MPI_LAND,    /* op       */
		   0,          /* root     */
		   MPI_COMM_WORLD /* comm  */);
  if (err != MPI_SUCCESS)
    goto end;

  //printf("Reduced\n");

end:
  delete[] block;
  delete[] displacements;
  delete[] displacements_transposed;
  delete[] counts;
  MPI_Type_free(&block_t);
  return res;
}
