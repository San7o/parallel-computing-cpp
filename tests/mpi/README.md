# MPI

## MPI Functions

```
MPI_Send
MPI_Recv
MPI_Bcast
MPI_Scatter
MPI_Reduce
MPI_Gather
MPI_Allreduce // Reduce + Bcast
MPI_Allgather
```

## Run the target

```bash
mpicxx <source code>
mpirun -np <number of processes> <target>
```
