# modern-c++template

This repo serves as a template for my C++ projects.

The template is composed of:

- [cmake](./CMakeLists.txt), [meson](./meson.build) and [bazel](https://bazel.build/) build files, already configured for:
    - testing / fuzzing / benchmarking
    - static and dynamic library build
    - gcc or clang option with c++23
    - [CPM](https://github.com/cpm-cmake/CPM.cmake) as cmake dependency manager
- [valFuzz](https://github.com/San7o/valFuzz) for testing, fuzzing and benchmarking
- [doxygen](./doxtgen.conf) documentation
- [clang-format](./.clang-format) settings
- [nix](./flake.nix) developement shell
- [cppcheck](https://cppcheck.sourceforge.io/) for static analysis 
- [LICENSE](./LICENSE), [CONTRIBUTING](./CONTRIBUTING.md) and other [git](./.gitattributes) files
- github workflow
- useful ready [commands](./Makefile)

## Building

### cmake
```bash
cmake -Bbuild
cmake --build build -j 4
```
### meson
```bash
meson setup build
ninja -C build
```
### bazel
```bash
bazel build //src:pc --sandbox_debug --verbose_failures
```

## Testing
```
cmake -Bbuild -DPC_BUILD_TESTS=ON
cmake --build build -j 4
./build/tests --help
```
The library uses [valFuzz](https://github.com/San7o/valFuzz) for testing
```c++
./build/tests              # run tests
./build/tests --fuzz       # run fuzzer
./build/tests --benchmark  # run benchmarks
```

## Documentation

The library uses doxygen for documentation, to build the html documentation run:
```
make docs
```

# Run on cluster:
Setup the `.bashrc` with module load + alias.
Do the same inside the `.bps` file.

In `hello.pbs`:
- set output files
- set maximum wall time
- change the working directory
- change the code to be compiled
Send a job to the queue:
```bash
qsum myProgram.pbs
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

## Flags
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

# TODO
- Use tennotl
