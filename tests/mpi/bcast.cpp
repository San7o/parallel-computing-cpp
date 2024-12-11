#include <mpi.h>
#include <cstdlib>
#include <cstdio>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int err;
  size_t message;
  if (world_rank == 0)
    message = 777;
  err = MPI_Bcast(&message, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS)
    {
      fprintf(stderr, "RANK: %d, Error sending message\n", world_rank);
    }
  fprintf(stdout, "RANK: %d, Message %d\n", world_rank, message);
  
  MPI_Finalize();
  return 0;
}
