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
#include <omp.h>

namespace pc
{

  
/*============================================*\
|                   BASELINE                   |
\*============================================*/

void matTranspose(float **M, float **T, tenno::size N);
void matTransposeHalf(float **M, float **T, tenno::size N);
void matTransposeColumns(float **M, float **T, tenno::size N);
void matTransposeCyclic(float *M, float *T, tenno::size N);
void matTransposeIntrinsic(float **mat_in, float **mat_out, size_t N);
void matTransposeIntrinsicCyclic(float *mat_in, float *mat_out, size_t N);


/*============================================*\
|                     MPI                      |
\*============================================*/

void matTransposeMPI(float *M, float *T, tenno::size N);
void matTransposeMPIBlock(float *M, float *T, tenno::size N);

// Used for debugging purposes
void matTransposeMPIBlockDebug(float *M, float *T, tenno::size N);

} // namespace pc
