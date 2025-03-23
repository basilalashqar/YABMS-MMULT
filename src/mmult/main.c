/* main.c
 *
 * Author: Khalid Al-Hawaj
 * Date  : 12 Nov. 2023
 * Edited By Basil Alashqar to optimize it for mmult benchmark
 * Date : 23 Mar. 2025
 *
 * This file is structured to call different implementations of the same
 * algorithm/microbenchmark. The file will allocate 3 output arrays: one
 * for the scalar naive implementation, one for the scalar optimized implementation,
 * and one for the vectorized implementation. As it stands the file will allocate and
 * initialize with random data two input arrays of type 'byte' (which represent matrices A and B).
 * To check correctness, the file allocates a 'ref' array; to calculate this 'ref' array,
 * the file invokes a ref_impl, which is supposed to be functionally correct and act as a reference
 * for the functionality. The file also adds a guard word at the end of the output arrays to check for
 * buffer overruns.
 *
 * The file will invoke each implementation n number of times. It will record the runtime
 * of _each_ invocation through the following Linux API:
 *    clock_gettime(), with the clk_id set to CLOCK_MONOTONIC
 * Then, the file will calculate the standard deviation and calculate an outlier-free
 * average by excluding runtimes that are larger than 2 standard deviations of the original average.
 */

/* Set features         */
#define _GNU_SOURCE

/* Standard C includes  */
/*  -> Standard Library */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
/*  -> Scheduling       */
#include <sched.h>
/*  -> Types            */
#include <stdbool.h>
#include <inttypes.h>
/*  -> Runtimes         */
#include <time.h>
#include <unistd.h>
#include <errno.h>

/* Include all implementations declarations */
#include "impl/ref.h"
#include "impl/naive.h"
#include "impl/opt.h"
#include "impl/vec.h"
#include "impl/para.h"

/* Include common headers */
#include "common/types.h"
#include "common/macros.h"

/* Include application-specific headers */
#include "include/types.h"

/* Default matrix dimensions */
int M_val = 16;
int N_val = 12;
int P_val = 8;

/* Default number of runs and standard-deviation threshold */
int nruns   = 10000;
int nstdevs = 3;

/* Default scheduling parameters */
int nthreads = 1;
int cpu      = 0;

/* Flag to indicate if matrices should be loaded from file */
bool load_data = false;

/* Helper function to load a binary file into a buffer */
void load_file(const char* filename, byte* buffer, size_t bytes) {
    FILE* fp = fopen(filename, "rb");
    if(fp == NULL) {
        printf("Error: Could not open file %s for reading\n", filename);
        exit(1);
    }
    size_t read = fread(buffer, 1, bytes, fp);
    if (read != bytes) {
        printf("Error: Expected to read %zu bytes from %s, but read %zu\n", bytes, filename, read);
        exit(1);
    }
    fclose(fp);
}

/* Helper function to poll for a file until its size reaches expected_size */
void wait_for_file_size(const char* filename, size_t expected_size, int timeout, int interval) {
    printf("Waiting for %s to reach %zu bytes...\n", filename, expected_size);
    int waited = 0;
    while (waited < timeout) {
        FILE* fp = fopen(filename, "rb");
        if (fp != NULL) {
            fseek(fp, 0L, SEEK_END);
            size_t size = ftell(fp);
            fclose(fp);
            if (size >= expected_size) {
                printf("%s is ready (size = %zu bytes).\n", filename, size);
                return;
            }
        }
        sleep(interval);
        waited += interval;
    }
    printf("Warning: %s did not reach the expected size within %d seconds.\n", filename, timeout);
}

int main(int argc, char** argv)
{
  /* Set the buffer for printf to NULL */
  setbuf(stdout, NULL);

  /* --- Parse arguments --- */
  /* Function pointers for implementations */
  void* (*impl_scalar_naive_ptr)(void* args) = impl_scalar_naive;
  void* (*impl_scalar_opt_ptr  )(void* args) = impl_scalar_opt;
  void* (*impl_vector_ptr      )(void* args) = impl_vector;
  void* (*impl_parallel_ptr    )(void* args) = impl_parallel;

  /* Chosen implementation */
  void* (*impl)(void* args) = NULL;
  const char* impl_str      = NULL;

  bool help = false;
  for (int i = 1; i < argc; i++) {
    /* Implementation selection */
    if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--impl") == 0) {
      assert(++i < argc);
      if (strcmp(argv[i], "naive") == 0) {
        impl = impl_scalar_naive_ptr; impl_str = "scalar_naive";
      } else if (strcmp(argv[i], "opt") == 0) {
        impl = impl_scalar_opt_ptr; impl_str = "scalar_opt";
      } else if (strcmp(argv[i], "vec") == 0) {
        impl = impl_vector_ptr; impl_str = "vectorized";
      } else if (strcmp(argv[i], "para") == 0) {
        impl = impl_parallel_ptr; impl_str = "parallelized";
      } else {
        impl = NULL; impl_str = "unknown";
      }
      continue;
    }
    /* New: Matrix dimensions */
    if (strcmp(argv[i], "--M") == 0) {
      assert(++i < argc);
      M_val = atoi(argv[i]);
      continue;
    }
    if (strcmp(argv[i], "--N") == 0) {
      assert(++i < argc);
      N_val = atoi(argv[i]);
      continue;
    }
    if (strcmp(argv[i], "--P") == 0) {
      assert(++i < argc);
      P_val = atoi(argv[i]);
      continue;
    }
    /* Option to load matrices from file */
    if (strcmp(argv[i], "--load") == 0) {
      load_data = true;
      continue;
    }
    /* Other parameters remain unchanged */
    if (strcmp(argv[i], "--nruns") == 0) {
      assert(++i < argc);
      nruns = atoi(argv[i]);
      continue;
    }
    if (strcmp(argv[i], "--nstdevs") == 0) {
      assert(++i < argc);
      nstdevs = atoi(argv[i]);
      continue;
    }
    if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--nthreads") == 0) {
      assert(++i < argc);
      nthreads = atoi(argv[i]);
      continue;
    }
    if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--cpu") == 0) {
      assert(++i < argc);
      cpu = atoi(argv[i]);
      continue;
    }
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      help = true;
      continue;
    }
  }
  if (help || impl == NULL) {
    if (!help) {
      if (impl_str != NULL) {
        printf("\nERROR: Unknown \"%s\" implementation.\n", impl_str);
      } else {
        printf("\nERROR: No implementation was chosen.\n");
      }
    }
    printf("\nUsage:\n");
    printf("  %s {-i | --impl} impl_str [Options]\n", argv[0]);
    printf("\nRequired:\n");
    printf("  -i | --impl      Available implementations = {naive, opt, vec, para}\n");
    printf("\nOptions:\n");
    printf("  --M              Number of rows for matrix A (default = %d)\n", M_val);
    printf("  --N              Number of columns for matrix A and rows for matrix B (default = %d)\n", N_val);
    printf("  --P              Number of columns for matrix B (default = %d)\n", P_val);
    printf("  --load           Load matrix A and B from A.bin and B.bin instead of random generation\n");
    printf("  -n | --nthreads  Set number of threads available (default = %d)\n", nthreads);
    printf("  -c | --cpu       Set the main CPU for the program (default = %d)\n", cpu);
    printf("  --nruns          Number of runs to the implementation (default = %d)\n", nruns);
    printf("  --nstdevs        Number of standard deviations to exclude outliers (default = %d)\n", nstdevs);
    printf("  -h | --help      Print this message\n\n");
    exit(help ? 0 : 1);
  }

  /* --- Scheduling --- */
  int nice_level = -20;
  printf("Setting up schedulers and affinity:\n");
  printf("  * Setting the niceness level:\n");
  do {
    errno = 0;
    printf("      -> trying niceness level = %d\n", nice_level);
    int __attribute__((unused)) ret = nice(nice_level);
  } while (errno != 0 && nice_level++);
  printf("    + Process has niceness level = %d\n", nice_level);
#if !defined(__APPLE__)
  printf("  * Setting up FIFO scheduling scheme and high priority ... ");
  pid_t pid = 0;
  int policy = SCHED_FIFO;
  struct sched_param param;
  param.sched_priority = sched_get_priority_max(policy);
  int res = sched_setscheduler(pid, policy, &param);
  if (res != 0) { printf("Failed\n"); } else { printf("Succeeded\n"); }
  printf("  * Setting up scheduling affinity ... ");
  cpu_set_t cpumask;
  CPU_ZERO(&cpumask);
  for (int i = 0; i < nthreads; i++) {
    CPU_SET((cpu + i) % nthreads, &cpumask);
  }
  res = sched_setaffinity(pid, sizeof(cpumask), &cpumask);
  if (res != 0) { printf("Failed\n"); } else { printf("Succeeded\n"); }
#endif
  printf("\n");

  /* --- Statistics --- */
  __DECLARE_STATS(nruns, nstdevs);

  /* --- Initialize Random --- */
  srand(0xdeadbeef);

  /* --- Datasets Allocation --- */
  /* For mmult:
       Matrix A: dimensions M_val x N_val
       Matrix B: dimensions N_val x P_val
       Output matrix R: dimensions M_val x P_val */
  size_t sizeA = (size_t)M_val * N_val * sizeof(float);
  size_t sizeB = (size_t)N_val * P_val * sizeof(float);
  size_t sizeR = (size_t)M_val * P_val * sizeof(float);

  byte* src0 = __ALLOC_INIT_DATA(byte, sizeA + 0);
  byte* src1 = __ALLOC_INIT_DATA(byte, sizeB + 0);
  byte* ref  = __ALLOC_INIT_DATA(byte, sizeR + 4);
  byte* dest = __ALLOC_DATA(byte, sizeR + 4);

  /* If --load option is provided, load matrices A and B from files */
  if(load_data) {
    printf("Loading matrix A from A.bin and matrix B from B.bin\n");
    load_file("A.bin", src0, sizeA);
    load_file("B.bin", src1, sizeB);
  } else {
    /* Otherwise, the __ALLOC_INIT_DATA already initializes with random data */
  }

  /* Set guards on output arrays */
  __SET_GUARD(ref, sizeR);
  __SET_GUARD(dest, sizeR);

  /* --- Generate Reference Data --- */
  args_t args_ref;
  args_ref.A = (float*) src0;
  args_ref.B = (float*) src1;
  args_ref.R = (float*) ref;
  args_ref.M = M_val;
  args_ref.N = N_val;
  args_ref.P = P_val;
  args_ref.cpu = cpu;
  args_ref.nthreads = nthreads;
  impl_ref(&args_ref);

  /* --- Execute Chosen Implementation --- */
  args_t args;
  args.A = (float*) src0;
  args.B = (float*) src1;
  args.R = (float*) dest;
  args.M = M_val;
  args.N = N_val;
  args.P = P_val;
  args.cpu = cpu;
  args.nthreads = nthreads;

  printf("Running \"%s\" implementation:\n", impl_str);
  printf("  * Invoking the implementation %d times .... ", nruns);
  for (int i = 0; i < nruns; i++) {
    __SET_START_TIME();
    for (int j = 0; j < 1; j++) {
      (*impl)(&args);
    }
    __SET_END_TIME();
    runtimes[i] = __CALC_RUNTIME();
  }
  printf("Finished\n");

  /* --- Verification --- */
  printf("  * Verifying results .... ");
  bool match = __CHECK_MATCH(ref, dest, sizeR);
  bool guard = __CHECK_GUARD(dest, sizeR);
  if (match && guard) {
    printf("Success\n");
  } else if (!match && guard) {
    printf("Fail, but no buffer overruns\n");
  } else if (match && !guard) {
    printf("Success, but failed buffer overruns check\n");
  } else if (!match && !guard) {
    printf("Failed, and failed buffer overruns check\n");
  }

  /* --- Additional: Write computed result to file and compare with Python file --- */
  {
    /* Write the computed result (dest) to a binary file */
    FILE* fp_out = fopen("computed.bin", "wb");
    if (fp_out == NULL) {
      printf("Error opening file for writing computed.bin\n");
    } else {
      size_t num_elems = sizeR / sizeof(float);
      fwrite(dest, sizeof(float), num_elems, fp_out);
      fclose(fp_out);
      printf("Computed result written to computed.bin\n");
    }

    /* Wait until computed.bin reaches the expected size */
    wait_for_file_size("computed.bin", sizeR, 10, 1);

    /* Now load a Python-generated reference file ("python_ref.bin") and compare */
    FILE* fp_py = fopen("python_ref.bin", "rb");
    if (fp_py == NULL) {
      printf("Error: Could not open python_ref.bin for reading\n");
    } else {
      float* py_data = (float*) malloc(sizeR);
      size_t expected_elems = sizeR / sizeof(float);
      size_t read_elems = fread(py_data, sizeof(float), expected_elems, fp_py);
      fclose(fp_py);
      if (read_elems != expected_elems) {
        printf("Error: Incomplete data read from python_ref.bin (expected %zu elements, got %zu)\n", expected_elems, read_elems);
      } else {
        int cmp_ok = 1;
        /* Relaxed tolerance for larger datasets */
        float tol = 1e-3;
        float max_diff = 0.0f;
        for (size_t i = 0; i < expected_elems; i++) {
          float diff = fabs(((float*)dest)[i] - py_data[i]);
          if (diff > max_diff)
            max_diff = diff;
          if (diff > tol) {
            cmp_ok = 0;
            /* Continue looping to record max_diff */
          }
        }
        if (cmp_ok)
          printf("Comparison with python_ref.bin: Success (max diff = %g)\n", max_diff);
        else
          printf("Comparison with python_ref.bin: Fail (max diff = %g)\n", max_diff);
      }
      free(py_data);
    }
  }

  /* --- Statistics and Outlier Analysis --- */
  uint64_t min = -1;
  uint64_t max = 0;
  uint64_t avg = 0, avg_n = 0;
  uint64_t std = 0, std_n = 0;
  int n_msked = 0;
  int n_stats = 0;
  for (int i = 0; i < nruns; i++)
    runtimes_mask[i] = true;
  printf("  * Running statistics:\n");
  do {
    n_stats++;
    printf("    + Starting statistics run number #%d:\n", n_stats);
    avg_n = 0;
    avg = 0;
    for (int i = 0; i < nruns; i++) {
      if (runtimes_mask[i]) {
        if (runtimes[i] < min) { min = runtimes[i]; }
        if (runtimes[i] > max) { max = runtimes[i]; }
        avg += runtimes[i];
        avg_n++;
      }
    }
    avg = avg / avg_n;
    std = 0;
    std_n = 0;
    for (int i = 0; i < nruns; i++) {
      if (runtimes_mask[i]) {
        std += ((runtimes[i] - avg) * (runtimes[i] - avg));
        std_n++;
      }
    }
    std = sqrt(std / std_n);
    n_msked = 0;
    for (int i = 0; i < nruns; i++) {
      if (runtimes_mask[i]) {
        if (runtimes[i] > avg) {
          if ((runtimes[i] - avg) > (nstdevs * std)) {
            runtimes_mask[i] = false;
            n_msked++;
          }
        } else {
          if ((avg - runtimes[i]) > (nstdevs * std)) {
            runtimes_mask[i] = false;
            n_msked++;
          }
        }
      }
    }
    printf("      - Standard deviation = %" PRIu64 "\n", std);
    printf("      - Average = %" PRIu64 "\n", avg);
    printf("      - Number of active elements = %" PRIu64 "\n", avg_n);
    printf("      - Number of masked-off = %d\n", n_msked);
  } while (n_msked > 0);
  printf("  * Runtimes (%s): ", __PRINT_MATCH(match));
  printf(" %" PRIu64 " ns\n", avg);

  /* --- Dump Runtimes to CSV --- */
  printf("  * Dumping runtime informations:\n");
  FILE * fp;
  char filename[256];
  strcpy(filename, impl_str);
  strcat(filename, "_runtimes.csv");
  printf("    - Filename: %s\n", filename);
  printf("    - Opening file .... ");
  fp = fopen(filename, "w");
  if (fp != NULL) {
    printf("Succeeded\n");
    printf("    - Writing runtimes ... ");
    fprintf(fp, "impl,%s", impl_str);
    fprintf(fp, "\n");
    fprintf(fp, "num_of_runs,%d", nruns);
    fprintf(fp, "\n");
    fprintf(fp, "runtimes");
    for (int i = 0; i < nruns; i++) {
      fprintf(fp, ", ");
      fprintf(fp, "%" PRIu64 "", runtimes[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "avg,%" PRIu64 "", avg);
    printf("Finished\n");
    printf("    - Closing file handle .... ");
    fclose(fp);
    printf("Finished\n");
  } else {
    printf("Failed\n");
  }
  printf("\n");

  /* --- Cleanup --- */
  free(src0);
  free(src1);
  free(dest);
  free(ref);

  __DESTROY_STATS();

  return 0;
}

