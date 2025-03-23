/* naive.c
 *
 * Author:
 * Date  :
 *
 * Description: Naïve matrix-matrix multiplication benchmark.
 */

/* Standard C includes */
#include <stdlib.h>
#include <stdio.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"

/* Naive Implementation */
#pragma GCC push_options
#pragma GCC optimize ("O1")

void* impl_scalar_naive(void* args)
{
    args_t* mmult_args = (args_t*) args;
    
    float* A = mmult_args->A;
    float* B = mmult_args->B;
    float* R = mmult_args->R;
    
    size_t M = mmult_args->M;
    size_t N = mmult_args->N;
    size_t P = mmult_args->P;

    // Naïve Matrix Multiplication Algorithm
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < P; j++) {
            R[i * P + j] = 0.0f;  // Initialize result matrix
            for (size_t k = 0; k < N; k++) {
                R[i * P + j] += A[i * N + k] * B[k * P + j];
            }
        }
    }

    return NULL;
}

#pragma GCC pop_options

