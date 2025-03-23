/* types.h
 *
 * Author: Khalid Al-Hawaj
 * Date  : 13 Nov. 2023
 * 
 * This file contains all required types declarations.
*/

#ifndef __INCLUDE_TYPES_H_
#define __INCLUDE_TYPES_H_

#include <stddef.h>

typedef struct {
  float* A;    // Input matrix A
  float* B;    // Input matrix B
  float* R;    // Output matrix R

  size_t M;    // Rows in A and R
  size_t N;    // Columns in A, Rows in B
  size_t P;    // Columns in B and R

  int cpu;     // CPU scheduling
  int nthreads; // Number of threads for parallel execution
} args_t;

#endif //__INCLUDE_TYPES_H_

