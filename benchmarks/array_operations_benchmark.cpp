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

#include <pc/ilp.hpp>
#include <valfuzz/valfuzz.hpp>

BENCHMARK(array_operations_benchmark, "array_operations")
{
    int *a = new (std::nothrow) int[SIZE];
    int *b = new (std::nothrow) int[SIZE];
    int *c = new (std::nothrow) int[SIZE];

    if (a == nullptr || b == nullptr || c == nullptr)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    /* Unoptimized code */
    RUN_BENCHMARK([a, b, c]() -> void {
        for (int j = 0; j < SIZE; ++j)
            a[j] = b[j] = c[j] = j;

        pc::array_operations(a, b, c);
        return;
    }());

    /* Optimized code */
    RUN_BENCHMARK([a, b, c]() -> void {
        for (int j = 0; j < SIZE; ++j)
            a[j] = b[j] = c[j] = j;

        pc::array_operations_optimized(a, b, c);
        return;
    }());

    delete[] a;
    delete[] b;
    delete[] c;
}
