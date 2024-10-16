#include <pc/pragma.hpp>

/**
 * Implicit parallelism
 */

void pc::original_loop(float *a, float *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        a[i] = b[i] * 2.5f + 10.0f;  // Simple linear transformation on each element
    }
}

// Optimized loop function using pragmas and loop unrolling
void pc::vectorized_loop(float *a, float *b, size_t size) {
	// Use vectorization and loop unrolling to optimize the loop
    #pragma GCC ivdep  // Ignore vector dependencies, carful with this pragma
    for (size_t i = 0; i < size; i += 4) {  // Unroll the loop by a factor of 4
        a[i] = b[i] * 2.5f + 10.0f;
        a[i + 1] = b[i + 1] * 2.5f + 10.0f;
        a[i + 2] = b[i + 2] * 2.5f + 10.0f;
        a[i + 3] = b[i + 3] * 2.5f + 10.0f;
    }
}
