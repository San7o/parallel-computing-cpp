# pc-first-assignment

> name: Giovanni Santini

> mat_id: 235441

## Index

- [About](#about)
- [Dependencies](#dependencies)
- [Building locally](#building_locally)
  - [Using cmake](#cmake)
  - [Using bazel](#bazel)
  - [Using meson](#meson)
- [Running locally](#running_locally)
- [Build and run on the cluster](#cluster)
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

### On NixOS

On NixOS, you can enter the developement
environment using [flake.nix](./flake.nix):

```bash
nix develop
```
<a name="building_locally"></a>
## Building locally

This project primarly supports building with `cmake >= 3.15.4` and
Its use is advised. Additionally, building with `bazel`
and `meson` is also supported for faster build time and better 
developement experience. Note that the instructions to run
the benchmarks on the cluster are specified using cmake.

<a name="cmake"></a>
### Using cmake

To build the full benchmark suite, run the following command:

```bash
git clone https://github.com/San7o/parallel-computing-cpp.git &&
cd parallel-computing-cpp &&
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

<a name="bazel"></a>
### Using bazel

If you decided to build with bazel, first clone the
repository with `--recursive` and then build the
target `first_assignment`:

```bash
git clone --recursive https://github.com/San7o/parallel-computing-cpp.git &&
bazel build //:first_assignment
```

The binaries will be generated in `bazel-bin`

<a name="meson"></a>
### Using meson

To build with meson, run the following:

```bash
git clone --recursive https://github.com/San7o/parallel-computing-cpp.git &&
meson setup build &&
ninja -C build
```

<a name="running_locally"></a>
## Running locally

The project depends on [valFuzz](https://github.com/San7o/valFuzz) which
provides testing, fuzzing and benchmarking functionalities.
To run the benchmarks, use the `--benchmark` flag. To get
more informations about possible flags, please consult
the help message with `--help`:

```c++
module load gcc91 &&
module load cmake-3.15.4 &&
./build/tests --benchmark          \
              --num-iterations 100 \
              --report report.txt
```

If you built optimized targets (on cmake using `PC_BUILD_OPTIMIZED_O{1,2,3}`)
you can easily run all the compiled targets with the [run-benchmarks.sh](./run-benchmarks.sh)
script:

```bash
./run-benchmarks.sh
```

To collect information about the system machine, you can use
the [get-info.sh](./get-info.sh) script:

```bash
./get-info.sh
```

<a name="cluster"></a>
## Build and run on the cluster

The complete `.pbs` file can be found in
[cluster/first-assignment/first-assignment.pbs](./cluster/first-assignment/first-assignment.pbs).
This file builds the binaries and executes them.
To submit the script for execution, run:

```bash
export PC_PBS_FILE=/home/giovanni.santini/cluster/parallel-computing-cpp/first-assignment/first-assignment.pbs &&
chmod +x $PC_PBS_FILE &&
qsub $PC_PBS_FILE
```

the script will generate reports in `benchamrks/plotting/reports/`
containing collected data in csv format.

<a name="additional_info"></a>
## Additional Information

<a name="analyze_data"></a>
### Analyze Data

All the graphs found in the report are generated
using seaborn in python. All the code can be
found in the `benchmarks/plotting` directory.

