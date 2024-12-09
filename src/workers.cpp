#include <mpi.h>
#include <unistd.h>
#include <stdio.h>

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
      exit(1);
    }

  char buff[10] = {};
  int err = MPI_Recv(&buff, 10, MPI_CHAR, 0, 0, MPI_COMM_WORLD, NULL);
  if (err != MPI_SUCCESS)
    {
      fprintf(stderr, "Error: MPI_Recv");
      exit(1);
    }
  fprintf(stdout, "WORKER %d: Received Message: %s", world_rank, buff);
  
  MPI_Finalize();
  return 0;
}