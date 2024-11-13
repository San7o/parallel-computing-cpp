#include <pc/transpose.hpp>
#include <tenno/ranges.hpp>

/*      i
         0  1  2  3
      
  j 0    0  1  2  3
    1    4  5  6  7
    2    8  9  10 11
    3    12 13 14 15
 */

/*
 * matTranspose
 *
 * For each element of the matrix with column i and row j,
 * the algorithm copies the value in the output matrix
 * in column j and row i.
 * Time Complexity: O(n^2)
 * Space Complexity: O(n^2)
 * Assumption: square matrix
 */
void pc::matTranspose(float **M, float **T, tenno::size N)
{
    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(N))
        {
            T[i][j] = M[j][i];
        }
    }
    return;
}

/*
 * matTransposeHalf
 *
 * An alternative algorithm to matTranspose is to iterate
 * over only the upper half of the matrix with respect
 * to the diagonal. This cuts in half the number of iterations
 * but the number of memory accesses (both read and writes)
 * do not change, as well as the time and space complexity.
 * Despite this, this algorithm improves from the previous
 * one by about 33%
 * Time Complexity: O(n^2)
 * Space Complexity: O(n^2)
 * Assumption: square matrix
 */
void pc::matTransposeHalf(float **M, float **T, tenno::size N)
{
    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(i, N))
        {
            T[i][j] = M[j][i];
            T[j][i] = M[i][j];
        }
    }
    return;
}

bool pc::checkSym(float **M, tenno::size N)
{
    for (auto i : tenno::range(N))
    {
        for (auto j : tenno::range(i, N))
        {
            if (M[i][j] != M[j][i])
            {
                return false;
            }
        }
    }
    return true;
}


/*
// Source: Stack overflow
void transpose(float *src, float *dst, const int N, const int M) {
    #pragma omp parallel for
    for(int n = 0; n<N*M; n++) {
        int i = n/N;
        int j = n%N;
        dst[n] = src[M*j + i];
    }
}

// With blocks
inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
}
*/
