#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <cstdlib>

/* World rank of this process */
/* should never be 0          */
int world_rank;

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0)
    {
      fprintf(stderr, "Error: worker should never be rank 0\n");
      MPI_Finalize();
      exit(1);
    }

  fprintf(stdout, "WORKER %d is listening...\n", world_rank);
  
  char buff[10] = {};
  int err = MPI_Bcast(&buff, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS)
    {
      fprintf(stderr, "Error: MPI_Recv");
      MPI_Finalize();
      exit(1);
    }
  fprintf(stdout, "WORKER %d: Received Message: %s\n", world_rank, buff);

  // TODO: Receive until STOP message
  
  MPI_Finalize();
  return 0;
}
