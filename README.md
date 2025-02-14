# pc-second-assignment

> name: Giovanni Santini

> mat_id: 235441

## Index

- [About](#about)
- [Dependencies](#dependencies)
- [Build and run on the cluster](#cluster)
- [Building manually](#building_manually)
- [Running manually](#running_manually)
- [Additional Information](#additional_info)
  - [Analyze data](#analyze_data)

<a name="about"></a>
## About

This repository contains the code for the first assignment of
the course, as well as the source for the report in latex and
all the necessary files to run the benchmarks on the cluster.

There is a high level view of the structure of this repo:

- `include/`: project's headers
- `src/`: algorithms implementation
- `benchmarks/`: all benchmarks
  - `plotting/`: python code to generate plots
- `cluster/`: pbs files to run on the cluster
- `tests/`: testing the algorithms
- `fuzz/`: fuzzing the algorithms
- `latex/`: report source code

<a name="dependencies"></a>
## Dependencies

The code is written in C++20 and it is tested with gcc-8.5.0,
9.1.0, 14.1.0 and 14.2.0 on a linux 6.10 and 6.12 and freeBSD 14.1
machine. Alternative/previous compilers or different operating
systems are not guaranteed to work.

This project depends on two libraries: [valfuzz](https://github.com/San7o/valFuzz.git)
and [tenno](https://github.com/San7o/tenno-tl.git).
- Valfuzz is used to run the benchmarks. The benchmarks are
  registered through a macro and are executed multile times, logging
  vaious information. To learn more please visit the library's
  github page.
- Tenno implements is a superset of the C++23 (and the upcoming 26)
  standard library and It provides useful data structures and
  functionalities to aid the developement of the benchmarks.
  An example of a feature used is compile time pseudo random
  number generation to initialize the input data.

Both dependencies are automatically downloaded by the build
system of your choice, thou cmake is primarly supported and Its
use is advised.

<a name="cluster"></a>
## Build and run on the cluster

The complete `.pbs` file can be found in
[cluster/first-assignment/first-assignment.pbs](./cluster/first-assignment/first-assignment.pbs).
This file builds the binaries and executes them.
To submit the script for execution, run the following on the cluster:

```bash
git clone https://github.com/San7o/parallel-computing-cpp.git &&
git checkout second-assignment &&
export PC_PBS_FILE=$HOME/parallel-computing-cpp/cluster/second-assignment/second-assignment.pbs &&
chmod +x $PC_PBS_FILE &&
qsub $PC_PBS_FILE
```

the script will generate reports in `$HOME/parallel-computing-cpp/benchamrks/plotting/reports/`
containing collected data in csv format.


<a name="building_manually"></a>
## Building manually

This project primarly supports building with `cmake >= 3.15.4`.
If you are on the cluster, please load the necessary modules first:
```bash
module load gcc91 &&
module load cmake-3.15.4 &&
module load mpich-3.2.1--gcc-9.1.0
```

To build the full benchmark suite, run the following command:

```bash
git clone https://github.com/San7o/parallel-computing-cpp.git &&
cd parallel-computing-cpp &&
git checkout second-assignment &&
cmake -Bbuild \
	  -D PC_BUILD_OPTIMIZED_O1=ON \
	  -D PC_BUILD_OPTIMIZED_O2=ON \
	  -D PC_BUILD_OPTIMIZED_O3=ON &&
cmake --build build \
      -j $(nproc)
```

Description of the arguments:

- `PC_BUILD_OPTIMIZED_O{1,2,3}=ON`: build a new target with specified
    optimization level
- `--build build`: output in the "build" directry
- `-j $(nproc)`: build with maximum number of threads. nproc
  is part of the gnu utils, if you don't have access to the
  program you can manually select a number or remove this
  argument completly.
 
Additional information:

Some options are enabled by default such as `PC_BUILD_OMP` for
omp support. You can build with clang by enabling
`PC_USE_CLANG` but the project is not fully compatible
with clang yet.

<a name="running_manually"></a>
## Running manually

The project depends on [valFuzz](https://github.com/San7o/valFuzz) which
provides testing, fuzzing and benchmarking functionalities.
To run the benchmarks, use the `--benchmark` flag. To get
more informations about possible flags, please consult
the help message with `--help`.
The benchmarks are composed of a master process and worker processes,
which will be build by default. You can run a benchmark like this:

```c++
module load gcc91 &&
module load cmake-3.15.4 &&
module load mpich-3.21--gcc-9.1.0 &&
export NUM_WORKERS=3 &&
export NUM_ITERATIONS=10 &&
export BUILD_DIR=build &&
mpirun -np 1 \
      ./$BUILD_DIR/tests \
      --benchmark \
      --num-iterations $NUM_ITERATIONS \
      --no-multithread \
      --report $OUTPUT_DIR/base_np4.txt \
		--reporter csv \
      : -np $NUM_WORKERS ./$BUILD_DIR/worker &&
unset NUM_WORKERS NUM_ITERATIONS BUILD_DIR
```

You could also run the full benchmarks by running the `.pbs` script:

```bash
chmod +x cluster/second-assignment/second-assignment.pbs &&
./cluster/second-assignment/second-assigmnet.pbs
```

There is a test `.bps` in the same directory to test the correctness
of the alogrithms.

To collect information about the system machine, you can use
the [get-info.sh](./get-info.sh) script:

```bash
./get-info.sh
```

<a name="additional_info"></a>
## Additional Information

<a name="analyze_data"></a>
### Analyze Data

All the graphs found in the report are generated
using seaborn in python. All the code can be
found in the `benchmarks/plotting` directory.

