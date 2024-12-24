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

BUILD_DIR="build"

if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi
if [ -d $BUILD_DIR ]; then
    if [ -f "$BUILD_DIR/tests" ]; then
        echo "Running regular tests with 2 workers..."
        mpirun -np 1 \
               ./$BUILD_DIR/tests \
	       --no-multithread \
               --verbose \
		: -np 1 ./$BUILD_DIR/worker
        echo "Running regular tests with 4 workers..."
        mpirun -np 1 \
               ./$BUILD_DIR/tests \
	       --no-multithread \
               --verbose \
		: -np 3 ./$BUILD_DIR/worker
        #echo "Running regular tests with 16 workers..."
        #mpirun -np 1 \
        #       ./$BUILD_DIR/tests \
	#       --no-multithread \
        #       --verbose \
        #	 : -np 15 ./$BUILD_DIR/worker
    fi
    if [ -f "$BUILD_DIR/tests_opt_o1" ]; then
        echo "Running tests o1..."
        mpirun -np 1 \
               ./$BUILD_DIR/tests_opt_o1 \
	       --no-multithread \
               --verbose \
		: -np 1 ./build/worker_opt_o1
    fi
    if [ -f "$BUILD_DIR/tests_opt_o2" ]; then
        echo "Running tests o2..."
        mpirun -np 1 \
               ./$BUILD_DIR/tests_opt_o2 \
	       --no-multithread \
               --verbose \
		: -np 1 ./build/worker_opt_o2
    fi
    if [ -f "$BUILD_DIR/tests_opt_o3" ]; then
        echo "Running tests o3..."
        mpirun -np 1 ./$BUILD_DIR/tests_opt_o3 \
	       --no-multithread \
               --verbose \
		: -np 1 ./build/worker_opt_o3
    fi
else
    echo "Please run the build script first"
fi
