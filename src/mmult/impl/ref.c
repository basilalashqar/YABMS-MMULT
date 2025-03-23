/* ref.c
 *
 * Author:
 * Date  :
 *
 * Description: Reference implementation of matrix-matrix multiplication.
 */

/* Standard C includes */
#include <stdlib.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"

/* Reference Implementation */
void* impl_ref(void* args)
{
    args_t* mmult_args = (args_t*) args;
    
    float* A = mmult_args->A;
    float* B = mmult_args->B;
    float* R = mmult_args->R;
    
    size_t M = mmult_args->M;
    size_t N = mmult_args->N;
    size_t P = mmult_args->P;
    
    /* Compute R = A x B in a triple-nested loop */
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < P; j++) {
            R[i * P + j] = 0.0f;  // initialize output element
            for (size_t k = 0; k < N; k++) {
                R[i * P + j] += A[i * N + k] * B[k * P + j];
            }
        }
    }
    
    return NULL;
}

