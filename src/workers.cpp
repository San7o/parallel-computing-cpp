#include <pc/transpose.hpp>
#include <pc/mpi.hpp>
#include <unistd.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>

/* World rank of this process */
/* should never be 0          */
int world_rank;

int main(int argc, char** argv)
{
  mpi::Init(&argc, &argv);
  mpi::Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0)
    {
      fprintf(stderr, "Error: worker should never be rank 0\n");
      mpi::Finalize();
      exit(1);
    }

  fprintf(stdout, "WORKER %d is listening...\n", world_rank);
  
  char func[10] = {};
  int err = mpi::Bcast(&func, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
  if (err != mpi::SUCCESS)
    {
      fprintf(stderr, "Error: MPI_Recv");
      mpi::Finalize();
      exit(1);
    }
  fprintf(stdout, "WORKER %d: Function: %s\n", world_rank, func);

  int N = {};
  do {
    err = mpi::Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); /* get N */
    if (err != mpi::SUCCESS)
    {
	fprintf(stderr, "Error: MPI_Bcast");
	mpi::Finalize();
	exit(1);
    }

    fprintf(stdout, "WORKER %d: Function: %s\n", world_rank, func);

    fprintf(stdout, "WORKER %d: N: %d\n", world_rank, N);
    fprintf(stdout, "WORKER %d: Is \"%s\" equal to \"Invert2\" ??\n", world_rank, func);
    if (strcmp(func, "Invert2") == 0)
	{
	fprintf(stdout, "WORKER %d: Inside Invert2\n", world_rank);
	float **mat1 = {};
	float **mat2 = {};
	pc::matTransposeMPIInvert2(mat1, mat2, N);
	}
    else {
	fprintf(stdout, "WORKER %d: No function detected\n", world_rank);
    }
  } while (N != -1);

  
  MPI_Finalize();
  return 0;
}
