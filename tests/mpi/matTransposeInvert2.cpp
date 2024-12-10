#include <pc/mpi.hpp>
#include <iostream>

int world_rank;

/* Executed by root */
void matTransposeMPIInvert2(float **M, float **T, size_t N)
{

  /* Create a new datatype */
  mpi::Datatype row_t;
  int err = mpi::Type_vector(N, N, 0, MPI_FLOAT, &row_t);
  if (err != mpi::SUCCESS)
    return;

  /* MPI Scatter */
  float row[N];
  err = mpi::Scatter(&M, N, row_t, &row, 1, row_t, 0, MPI_COMM_WORLD);
  if (err != mpi::SUCCESS)
    return;

  /* MPI Gather */
  
  return;
}

/* Executed by workers */
void matTransposeMPIInvert2Worker(float **M, float **T, size_t N)
{
  /* Create a new datatype */
  mpi::Datatype row_t;
  int err = mpi::Type_vector(N, N, 0, MPI_FLOAT, &row_t);
  if (err != mpi::SUCCESS)
    return;

  float row[N];
  err = mpi::Scatter(NULL, N, row_t, &row, 1, row_t, 0, MPI_COMM_WORLD);
  if (err != mpi::SUCCESS)
    return;

  std::cout << "WORKER " << world_rank << " Received: " << world_rank;
  for (int i = 0; i < N; ++i)
    std::cout << row[i] << " ";
  std::cout << std::endl;
}

int main(void)
{
  mpi::Init(NULL, NULL);

  mpi::Comm_rank(MPI_COMM_WORLD, &world_rank);

  constexpr size_t N = 16;
  float** mat1;
  float** mat2;
  if (world_rank == 0)
    {
	/* Initialize matices */
	mat1 = new float* [N];
	mat2 = new float* [N];
	for (size_t i = 0; i < N; ++i)
	    {
	    mat1[i] = new float[N];
	    mat2[i] = new float[N];
	    for (int j = 0; j < N; ++j)
	      mat1[i][j] = float(j * i);
	    }

	matTransposeMPIInvert2(mat1, mat2, N);

	/* Cleanup */
	for (size_t i = 0; i < N; ++i)
	{
	    delete[] mat1[i];
	    delete[] mat2[i];
	}
	delete[] mat1;
	delete[] mat2;
    }
  else
    {
      matTransposeMPIInvert2Worker(mat1, mat2, N);
    }
  mpi::Finalize();

  return 0;
}
