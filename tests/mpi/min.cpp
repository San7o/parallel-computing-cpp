/*
 * Write a program that calculates the minimum value across
 * all ranks using MPI_Allreduce. Each rank prints the results.
 */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int val = world_rank + 10;

  // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
  //                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
  int min;
  int err = MPI_Allreduce(&val, &min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS)
    {
      fprintf(stderr, "Error allreduce\n");
    }
  fprintf(stdout, "RANK: %d, Minimum value: %d\n", world_rank, min); 
  MPI_Finalize();
  return 0;
}
