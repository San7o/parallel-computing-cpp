#include <pc/transpose.hpp>
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>

/* World rank of this process */
/* should never be 0          */
namespace pc
{

int world_rank;
int world_size;

} // namespace pc

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pc::world_rank);
  if (pc::world_rank == 0)
    {
      fprintf(stderr, "Error: worker should never be rank 0\n");
      MPI_Finalize();
      exit(1);
    }
  MPI_Comm_size(MPI_COMM_WORLD, &pc::world_size);

  fprintf(stdout, "WORKER %d is listening...\n", pc::world_rank);
  
  size_t N = 0;
  while(true) {

    char func[10] = {};
    int err = MPI_Bcast(&func, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
    {
	fprintf(stderr, "Error: MPI_Recv");
	MPI_Finalize();
	exit(1);
    }

    if (strcmp(func, "") == 0) /* stop message */
    {
      fprintf(stderr, "WORKER %d: DONE\n", pc::world_rank);
      MPI_Finalize();
      exit(0);
    }

    err = MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); /* get N */
    if (err != MPI_SUCCESS)
    {
	fprintf(stderr, "Error: MPI_Bcast");
	MPI_Finalize();
	exit(1);
    }

    long unsigned int num_iterations;
    err = MPI_Bcast(&num_iterations, 1,
		     MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
    {
	fprintf(stderr, "Error: MPI_Bcast 2");
	MPI_Finalize();
	exit(1);
    }

    fprintf(stdout, "WORKER %d: Function: %s\n", pc::world_rank, func);
    fprintf(stdout, "WORKER %d: N: %ld\n", pc::world_rank, N);
    fprintf(stdout, "WORKER %d: Iterations: %ld\n", pc::world_rank,
	    num_iterations);

    float *mat1 = {};
    float *mat2 = {};
    if (strcmp(func, "Base") == 0)
    {
      for (unsigned long i = 0; i < num_iterations; ++i)
      {
	pc::matTransposeMPI(mat1, mat2, N);
      }
    }
    else if (strcmp(func, "Block") == 0)
    {
      for (unsigned long i = 0; i < num_iterations; ++i)
	pc::matTransposeMPIBlock(mat1, mat2, N);
    }
    else {
      fprintf(stdout, "WORKER %d: No function detected\n", pc::world_rank);
    }
  };
  
  MPI_Finalize();
  return 0;
}
