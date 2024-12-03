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
  MPI_Status status;
  if (world_rank == 0)
    {
  for (int i = 0; i < world_size; ++i)
    {
      err = MPI_Send((void*) &world_rank, 1, MPI_INT,
		     (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS)
	{
	  fprintf(stderr, "RANK: %d, Error sending message!\n", world_rank);
	}
      fprintf(stdout, "RANK: %d, Sent message!\n", world_rank);
    }
    }
  else
    {

  int message;
  // int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
  //              int source, int tag, MPI_Comm comm,
  //              MPI_Status *status)
  err = MPI_Recv((void*) &message, 1, MPI_INT,
		 (world_rank - 1) % world_size, 0, MPI_COMM_WORLD,
		 &status);
  if (err != MPI_SUCCESS)
    {
      fprintf(stderr, "RANK: %d, Error receiving the message!\n", world_rank);
      exit(1);
    }
  fprintf(stdout, "RANK: %d, Received message: %d\n", world_rank, message);
    }
  MPI_Finalize();
  return 0;
}
