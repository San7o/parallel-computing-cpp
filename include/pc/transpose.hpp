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

void matTranspose(float **M, float **T, tenno::size N);
void matTransposeHalf(float **M, float **T, tenno::size N);
void matTransposeColumns(float **M, float **T, tenno::size N);
void matTransposeCyclic(float *M, float *T, tenno::size N);


/*============================================*\
|              IMPLICIT PARALLELISM            |
\*============================================*/

// Vectorization (pragma simd)       DONE
// Loop Unrolling inner (manual)     DONE
// Loop Unrolling outer (manual)     DONE
// #pragma GCC ivdep                 ASSUMED IN SIMD
// branchless inner                  DONE
// branchless outer                  DONE

void matTransposeVectorization(float **M, float **T, tenno::size N);
void matTransposeUnrolledInner(float **M, float **T, tenno::size N);
void matTransposeUnrolledOuter(float **M, float **T, tenno::size N);

void matTransposeHalfVectorization(float **M, float **T, tenno::size N);
void matTransposeHalfUnrolledInner(float **M, float **T, tenno::size N);
void matTransposeHalfUnrolledOuter(float **M, float **T, tenno::size N);
void matTransposeCyclicUnrolled(float *M, float *T, tenno::size N);


/*============================================*\
|              EXPLICIT PARALLELISM            |
\*============================================*/

// omp                     DONE
// omp collapse            DONE
// different schedulers    DONE

void matTransposeOmp2(float **M, float **T, tenno::size N);
void matTransposeOmp4(float **M, float **T, tenno::size N);
void matTransposeOmp8(float **M, float **T, tenno::size N);
void matTransposeOmp16(float **M, float **T, tenno::size N);
void matTransposeOmp32(float **M, float **T, tenno::size N);
void matTransposeOmp64(float **M, float **T, tenno::size N);

void matTransposeOmp2Collapse(float **M, float **T, tenno::size N);
void matTransposeOmp4Collapse(float **M, float **T, tenno::size N);
void matTransposeOmp8Collapse(float **M, float **T, tenno::size N);
void matTransposeOmp16Collapse(float **M, float **T, tenno::size N);
void matTransposeOmp32Collapse(float **M, float **T, tenno::size N);
void matTransposeOmp64Collapse(float **M, float **T, tenno::size N);

void matTransposeOmp16SchedStatic(float **M, float **T, tenno::size N);
void matTransposeOmp16SchedDynamic(float **M, float **T, tenno::size N);
void matTransposeOmp16SchedGuided(float **M, float **T, tenno::size N);

} // namespace pc
