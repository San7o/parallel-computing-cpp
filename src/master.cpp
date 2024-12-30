#include <pc/transpose.hpp>
#include <pc/check_symm.hpp>
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>

namespace pc
{

int world_rank;
int world_size;

} // namespace pc

int main(int argc, char** argv)
{
  if (argc != 4)
  {
      fprintf(stdout, "Usage: %s <function> <size> <repetitions>", argv[0]);
      exit(1);
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pc::world_rank);
  if (pc::world_rank != 0)
  {
      fprintf(stderr, "Error: master should be rank 0\n");
      MPI_Finalize();
      exit(1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &pc::world_size);

  char func[10];
  strcpy(func, argv[1]);
  size_t N = atoi(argv[2]);
  long unsigned int num_iterations = atoi(argv[3]);

  fprintf(stdout, "MASTER: Function: %s\n", func);
  fprintf(stdout, "MASTER: N: %ld\n", N);
  fprintf(stdout, "MASTER: Iterations: %ld\n", num_iterations);

  int err = MPI_Bcast(&func, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS)
  {
	fprintf(stderr, "Error: MPI_Recv");
	MPI_Finalize();
	exit(1);
  }

  err = MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); /* get N */
  if (err != MPI_SUCCESS)
  {
	fprintf(stderr, "Error: MPI_Bcast");
	MPI_Finalize();
	exit(1);
  }

  err = MPI_Bcast(&num_iterations, 1,
		   MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS)
  {
	fprintf(stderr, "Error: MPI_Bcast 2");
	MPI_Finalize();
	exit(1);
  }

  float *mat1 = new float[N*N];
  float *mat2 = new float[N*N];
  for (size_t i = 0; i < N*N; ++i)
  {
      mat1[(i%N)*N + (i/N)] = float(i);
      mat1[(i/N)*N + (i%N)] = float(i);
  }
  
  if (strcmp(func, "Base") == 0)
  {
    for (unsigned long i = 0; i < num_iterations; ++i)
    {
	pc::matTransposeMPI(mat1, mat2, N);
    }
  }
  else if (strcmp(func, "BlockDbg") == 0)
  {
    for (unsigned long i = 0; i < num_iterations; ++i)
	pc::matTransposeMPIBlockDebug(mat1, mat2, N);
  }
  else if (strcmp(func, "Block") == 0)
  {
    for (unsigned long i = 0; i < num_iterations; ++i)
	pc::matTransposeMPIBlock(mat1, mat2, N);
  }
  else if (strcmp(func, "Sym") == 0)
  {
    for (unsigned long i = 0; i < num_iterations; ++i)
	pc::checkSymMPI(mat1, N);
  }
  else {
    fprintf(stdout, "MASTER %d: No function detected\n", pc::world_rank);
  }

  char end[10] = "";
  err = MPI_Bcast(&end, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS)
  {
	fprintf(stderr, "Error: MPI_Recv");
	delete [] mat1;
	delete [] mat2;
	MPI_Finalize();
	exit(1);
  }
  
  fprintf(stdout, "MASTER: DONE\n");
  
  delete [] mat1;
  delete [] mat2;
  MPI_Finalize();
  return 0;
}
