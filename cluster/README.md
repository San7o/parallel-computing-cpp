#the  Cluster

This directory contains `.pbs` files and additional
files necessary to correctly benchmark the algorithms
on the cluster. This file contains useful information
on this subject.

## Run on cluster

In each `.pbs` file:
- set output files
- set maximum wall time
- change the working directory
- change the code to be compiled

Send a job to the queue:
```bash
qsub myProgram.pbs
```
Check that the job is executing:
```bash
qstat -u user.name
```
Start an interactive session:
```bash
qsub -I -q short_cpuQ
```
Here we can compile and do everything

## Optimization Flags

Optimization levels:
```
-O0    # No optimization
-O1    # Basic optimization
-O2    # More advanced optimization
-O3    # Maximum optimization
-Ofast # Included _O3 optimizations plus aggressive math optimizations
```
Vectorization flags:
```
-ftree-vectorize      # Automatic loop vectorization SIMD
-ffast-math           # Aggressive floating point optimizations
-fopt-info-vec or -fopt-info-vec-optimized # output detailed optimization reports
-fopt-info-vec-missed # Reports on missed optimization
-fopt-info-vec-all    # Reports all
```
Target-Specific Optimization Flags:
```
-march=native            # Enables architecture-specific optimizations
-funroll-loops           # forces the compiler to unroll loops
-fprefetch-loop-arrays   # Enables software prefetching of array elements
```

## Cache analysis

```bash
valgrind --tool=cachegrind ./executable
cg_annotate cachegrind.out.<pid>
```

## Load Modules

```bash
module avail  # list modules
module load <your module>
```

## Perf

List events:
```
perf list
```

Create a report:
```bash
perf record [-e <event>,...] <command>
```

Read the report:
```bash
perf report
```

Get more statistics:
```bash
perf stat <command>
```
