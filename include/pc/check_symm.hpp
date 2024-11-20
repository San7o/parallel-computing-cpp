/*
 * MIT License
 *
 * Copyright (c) 2024 Giovanni Santini

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

#pragma once

#include <tenno/types.hpp>

namespace pc
{


/*============================================*\
|                   BASELINE                   |
\*============================================*/

bool checkSym(float **M, tenno::size N);


/*============================================*\
|              IMPLICIT PARALLELISM            |
\*============================================*/

// Vectorization (pragma simd)       DONE
// Loop Unrolling inner (manual)     DONE
// Loop Unrolling outer (manual)     DONE
// #pragma GCC ivdep                 ASSUMED IN SIMD
// branchless inner                  DONE
// branchless outer                  DONE

bool checkSymVectorization(float **M, tenno::size N);
bool checkSymUnrollingInner(float **M, tenno::size N);
bool checkSymUnrollingOuter(float **M, tenno::size N);
bool checkSymUnrollingOuterNoBranch(float **M, tenno::size N);
bool checkSymUnrollingInnerNoBranch(float **M, tenno::size N);


/*============================================*\
|              EXPLICIT PARALLELISM            |
\*============================================*/

// omp                     DONE
// omp collapse            DONE
// different schedulers    DONE

bool checkSymOmp2(float **M, tenno::size N);
bool checkSymOmp4(float **M, tenno::size N);
bool checkSymOmp8(float **M, tenno::size N);
bool checkSymOmp16(float **M, tenno::size N);
bool checkSymOmp32(float **M, tenno::size N);
bool checkSymOmp64(float **M, tenno::size N);

bool checkSymOmp2Collapse(float **M, tenno::size N);
bool checkSymOmp4Collapse(float **M, tenno::size N);
bool checkSymOmp8Collapse(float **M, tenno::size N);
bool checkSymOmp16Collapse(float **M, tenno::size N);
bool checkSymOmp32Collapse(float **M, tenno::size N);
bool checkSymOmp64Collapse(float **M, tenno::size N);

bool checkSymOmp16SchedStatic(float **M, tenno::size N);
bool checkSymOmp16SchedDynamic(float **M, tenno::size N);
bool checkSymOmp16SchedGuided(float **M, tenno::size N);

} // pc
