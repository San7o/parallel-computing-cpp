/*
 * MIT License
 *
 * Copyright (c) 2024 Giovanni Santini
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include <pc/pragma.hpp>
#include <valfuzz/valfuzz.hpp>

BENCHMARK(pragma_mebchmark, "loop")
{
    size_t SIZE = 100000;
    float *a = new float[SIZE];
    float *b = new float[SIZE];

	auto run_original_loop = [a, b](size_t size) -> void {
        for (size_t j = 0; j < size; ++j)
            a[j] = b[j] * 2.0f;

        pc::original_loop(a, b, size);
        return;
    };

    RUN_BENCHMARK(100, run_original_loop(100));
    RUN_BENCHMARK(1000, run_original_loop(1000));
    RUN_BENCHMARK(10000, run_original_loop(10000));
    RUN_BENCHMARK(100000, run_original_loop(100000));

    delete[] a;
    delete[] b;
}

BENCHMARK(pragma_benchmark_vectorized, "vectorized_loop")
{
    size_t SIZE = 100000;
    float *a = new float[SIZE];
    float *b = new float[SIZE];

    auto run_vectorized_loop = [a, b](size_t size) -> void {
        for (size_t j = 0; j < size; ++j)
            a[j] = b[j] * 2.0f;

        pc::vectorized_loop(a, b, size);
        return;
    };

    RUN_BENCHMARK(100, run_vectorized_loop(100));
    RUN_BENCHMARK(1000, run_vectorized_loop(1000));
    RUN_BENCHMARK(10000, run_vectorized_loop(10000));
    RUN_BENCHMARK(100000, run_vectorized_loop(100000));

    delete[] a;
    delete[] b;
}
