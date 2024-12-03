/*
 * Write a program that calculates the minimum value across
 * all ranks using MPI_Allreduce. Each rank prints the results.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>  /* exit */

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  fprintf(stdout, "World size: %d\n", world_size);
  int val = world_rank + 10;

  int buff[] = { 0, 1, 2 };
  for (int i = 0; i < 3; ++i)
    buff[i] = buff[i] * (world_rank + 1);
  int rec[3 * world_size] = {};

  // int MPI_Alltoall(const void *sendbuf, int sendcount,
  //	   MPI_Datatype sendtype, void *recvbuf, int recvcount,
  //       MPI_Datatype recvtype, MPI_Comm comm)
  int err = MPI_Alltoall(buff, 3, MPI_INT,
			 rec, 3, MPI_INT, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS)
    {
      fprintf(stdout, "Error Alltoall\n");
      MPI_Finalize();
      exit(1);
    }

  fprintf(stdout, "RANK: %d, Output: \n", world_rank);
  for (int j = 0; j < 3; ++j)
    fprintf(stdout, "RANK %d, %d ", world_rank, rec[world_rank*3+j]);
  fprintf(stdout, "\n");
  
  MPI_Finalize();
  return 0;
}
