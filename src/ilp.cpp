/**
 * # Implicit ILP
 *
 * ILP is a measure of how many individual instructions in a program can be
 * processed simultaneously.
 *
 * ## Data dependencies
 *
 * ### True Dependency
 * ```cpp
 * a = b + c;
 * d = a * 2;
 * ```
 *
 * ### Anti dependency
 * ```cpp
 * b = a + 1;
 * a = 5;
 * ```
 *
 * ### Output dependency
 * ```cpp
 * a = 10;
 * a = 20;
 * ```
 *
 * ## Control dependencies
 * ```cpp
 * if (condition)
 *    a = 10;
 * ```
 *
 * # Loop unrolling
 * Original loop:
 * ```cpp
 * for (int i = 0; i < n; ++i)
 *   a[i] = b[i] + c[i];
 * ```
 * Unrolled loop:
 * ```cpp
 * for ( int i = 0; i < n ; i += 4) {
 *   a [ i ] = b [ i ] + c [ i ];
 *   a [ i +1] = b [ i +1] + c [ i +1];
 *   a [ i +2] = b [ i +2] + c [ i +2];
 *   a [ i +3] = b [ i +3] + c [ i +3];
 * }
 * ```
 *
 * # Instruction Reordering
 * Original code:
 * ```cpp
 * a = x + y;
 * b = a + z;
 * c = w * k;
 * ```
 * Reordered code:
 * ```cpp
 * a = x + y;
 * c = w * k;
 * b = a + z;
 * ```
 *
 * # Exercise:
 */
#include <pc/ilp.hpp>
void pc::array_operations(int *a, int *b, int *c)
{
    for (int i = 1; i < SIZE; i++)
    {
        a[i] = b[i] + b[i - 1]; // Instruction 1: True dependency on b[i - 1]
        b[i] = a[i]
               * 2; // Instruction 2: Anti dependency (Write After Read on b[i])
        c[i] = a[i] + b[i - 1]; // Instruction 3: Output dependency (Write After
                                // Write on a[i])
    }
}
/**
 * Can we unroll the loop?
 */

void pc::array_operations_optimized(int *a, int *b, int *c)
{
    for (int i = 1; i < SIZE - 4; i+=4)
    {
        a[i] = b[i] + b[i-1];
        a[i+1] = b[i+1] + b[i];
        a[i+2] = b[i+2] + b[i+1];
        a[i+3] = b[i+3] + b[i+2];

        b[i] = b[i] + b[i-1];
        b[i+1] = b[i+1] + b[i];
        b[i+2] = b[i+2] + b[i+1];
        b[i+3] = b[i+3] + b[i+2];

        c[i] = b[i] + 2 * b[i-1];
        c[i+1] = b[i+1] + 2 * b[i];
        c[i+2] = b[i+2] + 2 * b[i+1];
        c[i+3] = b[i+3] + 2 * b[i+2];
    }
}
