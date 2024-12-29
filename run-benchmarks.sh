#! /bin/sh

# MIT License
#
# Copyright (c) 2024 Giovanni Santini
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

NUM_ITERATIONS=1
BUILD_DIR="build"
OUTPUT_DIR=benchmarks/plotting/reports

if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi
if [ -d $BUILD_DIR ]; then
    if [ -f "$BUILD_DIR/tests" ]; then
        echo "Running regular benchmarks..."
        mpirun -np 1 ./$BUILD_DIR/tests \
                        --benchmark \
                        --num-iterations $NUM_ITERATIONS \
                        --no-multithread \
                        --reporter csv \
		: -np 1 ./build/worker
    fi
    if [ -f "$BUILD_DIR/tests_opt_o1" ]; then
        echo "Running optimized benchmarks..."
        mpirun -np 1 ./$BUILD_DIR/tests_opt_o1 \
                        --benchmark \
                        --num-iterations $NUM_ITERATIONS \
                        --no-multithread \
                        --reporter csv
    fi
    if [ -f "$BUILD_DIR/tests_opt_o2" ]; then
        echo "Running optimized benchmarks..."
        mpirun -np 1 ./$BUILD_DIR/tests_opt_o2 \
                        --benchmark \
                        --num-iterations $NUM_ITERATIONS \
                        --no-multithread \
                        --reporter csv
    fi
    if [ -f "$BUILD_DIR/tests_opt_o3" ]; then
        echo "Running aggressive optimized benchmarks..."
        mpirun -np 1 ./$BUILD_DIR/tests_opt_o3 \
                        --benchmark \
                        --num-iterations $NUM_ITERATIONS \
                        --no-multithread \
                        --reporter csv
    fi
else
    echo "Please run the build script first"
fi
