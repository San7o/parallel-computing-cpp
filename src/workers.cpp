#include <pc/transpose.hpp>
#include <pc/mpi.hpp>
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
  mpi::Init(&argc, &argv);
  mpi::Comm_rank(MPI_COMM_WORLD, &pc::world_rank);
  if (pc::world_rank == 0)
    {
      fprintf(stderr, "Error: worker should never be rank 0\n");
      mpi::Finalize();
      exit(1);
    }
  mpi::Comm_size(MPI_COMM_WORLD, &pc::world_size);

  fprintf(stdout, "WORKER %d is listening...\n", pc::world_rank);
  
  char func[10] = {};
  int err = mpi::Bcast(&func, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
  if (err != mpi::SUCCESS)
    {
      fprintf(stderr, "Error: MPI_Recv");
      mpi::Finalize();
      exit(1);
    }

  size_t N = 0;
  while(true) {
    err = mpi::Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); /* get N */
    if (err != mpi::SUCCESS)
    {
	fprintf(stderr, "Error: MPI_Bcast");
	mpi::Finalize();
	exit(1);
    }

    if (N == 0)
      break;

    fprintf(stdout, "WORKER %d: Function: %s\n", pc::world_rank, func);
    fprintf(stdout, "WORKER %d: N: %ld\n", pc::world_rank, N);


    if (strcmp(func, "Base") == 0)
    {
	float *mat1 = {};
	float *mat2 = {};
	pc::matTransposeMPI(mat1, mat2, N);
    }
    else {
      fprintf(stdout, "WORKER %d: No function detected\n", pc::world_rank);
    }
  };

  
  MPI_Finalize();
  return 0;
}
