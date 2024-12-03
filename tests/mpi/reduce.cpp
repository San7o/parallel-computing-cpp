#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int res;
  int tot;
  // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
  //               MPI_Datatype datatype, MPI_Op op, int root,
  //               MPI_Comm comm)
  res = MPI_Reduce(&world_rank, &tot, 1, MPI_INT, MPI_SUM,
		   0, MPI_COMM_WORLD);
  if (res != MPI_SUCCESS)
    {
      fprintf(stderr, "Error reducing!\n");
    }
  if (world_rank == 0)
    {
      fprintf(stdout, "Tot rank: %d\n", tot);
    }

  MPI_Finalize();
  return 0;
}
