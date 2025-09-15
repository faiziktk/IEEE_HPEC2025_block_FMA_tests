/* Test numerical features of CUDA wmma instruction.

   This test bench is based on the code in
   https://github.com/north-numerical-computing/tensor-cores-numerical-behavior

   Reference:
     Faizan A Khattak and Mantas Mikaitis,
     Generalized Methodology for Determining Numerical Features of Hardware
     Floating-Point Matrix Multipliers: Part I.
     Accepted for 29th Annual IEEE High Performance Extreme Computing. Sep. 2025.
 */

#include <assert.h>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <bitset>
#include <mma.h>
#include <iomanip>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace nvcuda;

/* Chose the input format of the tensor core (wmma)
   by ucommenting one of the three options. */

//#define IN_FORMAT_BF16
//#define IN_FORMAT_FP16
#define IN_FORMAT_TF32

/* Set wmma shape */
#define M 16
#define N 16
int pout = 24;

/* Base on the input format, set up various parameters. */
#ifdef IN_FORMAT_BF16
  #define IN_FORMAT nv_bfloat16
  #define WMMA_IN_FORMAT nv_bfloat16
  #define CONVERSION_OP  __float2bfloat16
  int pin = 8;
  #define K 16
#elif defined(IN_FORMAT_FP16)
  #define IN_FORMAT half
  #define WMMA_IN_FORMAT half
  #define CONVERSION_OP  __float2half
  int pin = 11;
  int pout16 = 11;
  #define K 16
#elif defined(IN_FORMAT_TF32)
  #define IN_FORMAT float
  #define WMMA_IN_FORMAT wmma::precision::tf32
  int pin = 11;
  #define CONVERSION_OP
  #define K 8
#endif

/* Set up the output format */
#define OUT_FORMAT float


/****************************************************
 * Memory management and wmma::mma_sync() interface *
 ****************************************************/

/* Set the entries of host arrays to zero. */
template <typename returntype>
void host_reset(IN_FORMAT* a, IN_FORMAT* b, returntype* c) {
  memset(a, 0, 16 * 16 * sizeof(IN_FORMAT));
  memset(b, 0, 16 * 16 * sizeof(IN_FORMAT));
  memset(c, 0, 16 * 16 * sizeof(returntype));
}


/* Compute C += A*B, where A, B, and C are MxN matrices.
   The matrix C is initialized to 0 when `init` is true. */
template <typename returntype>
__global__ void wmma_ker(IN_FORMAT* a, IN_FORMAT* b,
                         returntype* c, bool init) {

  // Declare fragments.
  wmma::fragment<wmma::matrix_a, M, N, K, WMMA_IN_FORMAT,
    wmma::row_major> a_fragment;
  wmma::fragment<wmma::matrix_b, M, N, K, WMMA_IN_FORMAT,
    wmma::col_major> b_fragment;
  wmma::fragment<wmma::accumulator, M, N, K, returntype> c_fragment;

  // Load input matrices and initialize output (if required).
  wmma::load_matrix_sync(a_fragment, a, N);
  wmma::load_matrix_sync(b_fragment, b, M);
  if (init)
    wmma::fill_fragment(c_fragment, 0.0f);
  else
    wmma::load_matrix_sync(c_fragment, c, N, wmma::mem_col_major);

  // Multiply
  wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

  // Store the output
  wmma::store_matrix_sync(c, c_fragment, N, wmma::mem_col_major);
}


/* Copy data from host to device, perform the operation, and copy result back to
   host. */
template <typename returntype>
void wmma_init_run(IN_FORMAT* h_a, IN_FORMAT* h_b, returntype* h_c,
                   IN_FORMAT* d_a, IN_FORMAT* d_b, returntype* d_c,
                   bool init) {

  // Copy input from host to device.
  cudaMemcpy(d_a, h_a, 16 * 16 * sizeof(IN_FORMAT), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, 16 * 16 * sizeof(IN_FORMAT), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, 16 * 16 * sizeof(returntype), cudaMemcpyHostToDevice);

  // Perform matrix multiplication.
  wmma_ker << <1, 32 >> > (d_a, d_b, d_c, init);

  // Copy result from device to host.
  cudaMemcpy(h_c, d_c, 16 * 16 * sizeof(returntype), cudaMemcpyDeviceToHost);
}


/**********************
 * Printing functions *
 **********************/
void printheader(FILE* outfile, const char* string) {
  fprintf(outfile,
          "+--------------------------------------------------------------+\n");
  fprintf(outfile, "| %-60s |\n", string);
  fprintf(outfile,
          "+--------------------------------------------------------------+\n");
}


/***************
 * EXPERIMENTS *
 ***************/
int main(int argc, char** argv) {

  IN_FORMAT* h_a, * h_b, * h16_c, * d16_a, * d16_b,  * d16_c;
  OUT_FORMAT* d_c, * h_c;
  h_a = new IN_FORMAT[16 * 16];
  h_b = new IN_FORMAT[16 * 16];
  h_c = new OUT_FORMAT[16 * 16];
  h16_c = new IN_FORMAT[M * N];

  cudaMalloc(&d16_a, 16 * 16 * sizeof(IN_FORMAT));
  cudaMalloc(&d16_b, 16 * 16 * sizeof(IN_FORMAT));
  cudaMalloc(&d16_c, M * N * sizeof(IN_FORMAT));
  cudaMalloc(&d_c, 16 * 16 * sizeof(OUT_FORMAT));

  FILE* outfile = stdout;

  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               1: Extra Alignment Bit test: (one and two bits)
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  int j = 0, neab = 0;
  printheader(outfile, "Feature 1. Extra Alignment Bits Determination");
  host_reset(h_a, h_b, h_c);
  /* Single bit test existence */
  h_a[0] = CONVERSION_OP(ldexp(1, (-pout + j)));
  h_b[0] = CONVERSION_OP(1);
  h_a[1] = CONVERSION_OP(ldexp(1, (-pout + j)));
  h_b[1] = CONVERSION_OP(1);
  h_c[0] = pow(2, j);
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  if (h_c[0] == (pow(2, j) + 2 * ldexp(1, -pout + j))) {
    printf("-> One bit existence test: Passed\n");
    neab = 1;
  }
  else {
    printf("-> One bit existence test: Failed\n");
  }

  /* Two bits test existence */
  h_a[0] = CONVERSION_OP(ldexp(1, (-pout + j)));
  h_b[0] = CONVERSION_OP(1);

  h_a[1] = CONVERSION_OP(ldexp(1, (-pout + j)));
  h_b[1] = CONVERSION_OP(ldexp(1, -1));
  h_a[2] = CONVERSION_OP(ldexp(1, (-pout + j)));
  h_b[2] = CONVERSION_OP(ldexp(1, -1));

  h_c[0] = pow(2, j);
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  if (h_c[0] == (pow(2, j) + 2 * ldexp(1, -pout + j))) {
    printf("-> Two bits existence test: Passed\n");
    neab = 2;
  }
  else {
    printf("-> Two bits existence test: Failed\n");
  }


  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               2: Extra Carry Bits (Iterative Approach)
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  /*Assuming the FMA size is less than permitted max shared dimension of tc*/
  int k = 2, necb = 0, NFMA = 0;
  OUT_FORMAT d_pos = 0,d_neg=0;
  while (k < 15) {
    host_reset(h_a, h_b, h_c);
    h_c[0] = 1 + ldexp(1, -pout + 1);
    h_a[0] = CONVERSION_OP(1);
    h_b[0] = CONVERSION_OP(1);
    h_b[k - 1] = h_b[0];
    h_a[k - 1] = CONVERSION_OP(ldexp(1, -pout + 1));
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    d_pos = h_c[0];
    // negative axis
    host_reset(h_a, h_b, h_c);
    h_c[0] = -1 - ldexp(1, -pout + 1);
    h_a[0] = CONVERSION_OP(-1);
    h_b[0] = CONVERSION_OP(1);
    h_b[k - 1] = h_b[0];
    h_a[k - 1] = CONVERSION_OP(-ldexp(1, -pout + 1));
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    d_neg = h_c[0];
    if (d_pos != (2 + ldexp(1, -pout + 2)) && abs(d_neg) !=
        (2 + ldexp(1, -pout + 2))) {
      NFMA = k - 1;
      break;
    }
    necb = floor(log2(k*(2-ldexp(1,-pin+1))));
    k = k + 1;
  }
  printheader(outfile,
    "Feature 2. Extra carry bits determination fp16/bf16 inputs");
  printf("Number of extra bits detected are: %d\n", necb);

  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               3: FMA Size (Iterative Approach)
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  printheader(outfile, "Feature 3. FMA size for fp16/bf16 inputs");
  printf("The FMA size is: %d\n", NFMA);


  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               4: Normalisation Pattern Within A Block FMA
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  /* eab=1, ecb=1; in this case, */
  printheader(outfile, "Feature 4. Normalisation pattern in BFMA");
  int t = 3; // for example, from HPEC-25 paper
  host_reset(h_a, h_b, h_c);
  h_a[0] = CONVERSION_OP(ldexp(1, -pout + t) + ldexp(1, -pout));
  h_a[1] = CONVERSION_OP(ldexp(1, -pout + t) + ldexp(1, -pout));
  h_b[0] = CONVERSION_OP(1);
  h_b[1] = CONVERSION_OP(1);
  h_c[0] = 1 - (ldexp(1, -pout + t));
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  if (h_c[0] == (1 + ldexp(1, -pout + t) + ldexp(1, -pout + 1))) {
    printf("-> Delayed/Late normalisation\n");
  }
  else if (h_c[0] == (1 + ldexp(1, -pout + t))) {
    printf("-> Immediate normalisation with RD/RN/TRC/RZ\n");
  }
  else if (h_c[0] == (1 + ldexp(1, -pout + t) + ldexp(1, -pout + 2))) {
    printf("-> Immediate normalisation with RU\n");
  }
  else {
    printf("-> This test is not working!\n");
  }


  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               5: Rounding Mode In a BFMA
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  /* Assuming FMA size is 3 or more for this case */
  j = 0; // any integer within max min limit definde by exponent
  OUT_FORMAT h_c_pos = 0, h_c_neg = 0, h_c_temp = 0;
  printheader(outfile,
    "Feature 5. Rounding Mode for final output in SP of a BFMA");
  h_c[0] = ldexp(1, j) + ldexp(1, -pout + j + 1) + ldexp(1, -pout + j + 2);
  h_c_temp = h_c[0];
  h_a[0] = CONVERSION_OP(ldexp(1, j));
  h_a[1] = h_a[0];
  h_a[2] = h_a[0];
  h_b[0] = CONVERSION_OP(1);
  h_b[1] = h_b[0]; h_b[2] = h_b[1];
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  h_c_pos = h_c[0];
  // negative axis
  h_c[0] = -h_c_temp;
  h_a[0] = -h_a[0];
  h_a[1] = h_a[0];
  h_a[2] = h_a[0];
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  h_c_neg = h_c[0];

  if (h_c_pos == ldexp(1, j + 2) & h_c_neg == -ldexp(1, j + 2)) {
    printf("-> Round towards zero/truncation\n");
  }
  else if (h_c_pos == (ldexp(1, j + 2) +
                       ldexp(1, -pout + j + 3)) &
           h_c_neg == -(ldexp(1, j + 2)
                        + ldexp(1, -pout + j + 3))) {
    printf("-> Round to nearest (even)\n");
  }
  else if (h_c_pos == (ldexp(1, j + 2) +
                       ldexp(1, -pout + j + 3)) &
           h_c_neg == -(ldexp(1, j + 2))) {
    printf("-> Round up!\n");
  }
  else if (h_c_pos == (ldexp(1, j + 2)) &
           h_c_neg == -(ldexp(1, j + 2) +
                        ldexp(1, -pout + j + 3))) {
    printf("-> Round down\n");
  }
  else {
    printf("-> This test is not working!\n");
  }


#ifdef IN_FORMAT_FP16
  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               6: Rounding Mode in a BFMA for FP16 Output
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  /* Assuming FMA size is 3 or more for this case */
  j = 0; // any integer within max min limit definde by exponent
  half h16_c_pos = 0, h16_c_neg = 0;
  printheader(outfile,
    "Feature 6. Rounding Mode for final output in fp16 of a BFMA");
  host_reset(h_a, h_b, h16_c);

  h16_c[0] = __float2half(ldexp(1, j) + ldexp(1, -pout16 + j + 1) +
                          ldexp(1, -pout16 + j + 2));
  h_a[0] = __float2half(ldexp(1, j));
  h_a[1] = h_a[0];
  h_a[2] = h_a[0];
  h_b[0] = __float2half(1);
  h_b[1] = h_b[0]; h_b[2] = h_b[1];
  wmma_init_run(h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
  h16_c_pos = h16_c[0];
  // negative axis
  host_reset(h_a, h_b, h16_c);
  h16_c[0] = __float2half(-ldexp(1, j) - ldexp(1, -pout16 + j + 1) -
                          ldexp(1, -pout16 + j + 2));
  h_a[0] = __float2half(-ldexp(1, j));
  h_a[1] = h_a[0];
  h_a[2] = h_a[0];
  h_b[0] = __float2half(1);
  h_b[1] = h_b[0]; h_b[2] = h_b[1];
  wmma_init_run(h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
  h16_c_neg = h16_c[0];


  if (h16_c_pos == __float2half(ldexp(1, j + 2)) & h16_c_neg ==
      __float2half (- ldexp(1, j + 2))) {
    printf("-> Round towards zero/truncation\n");
  }
  else if (h16_c_pos == __float2half(ldexp(1, j + 2) +
                                     ldexp(1, -pout16 + j + 3)) & h16_c_neg ==
           __float2half(-ldexp(1, j + 2) - ldexp(1, -pout16 + j + 3))) {
    printf("-> Round to nearest (even)\n");
  }
  else if (h16_c_pos == __float2half(ldexp(1, j + 2) +
                                     ldexp(1, -pout16 + j + 3)) & h16_c_neg ==
           __float2half(-ldexp(1, j + 2))) {
    printf("-> Round up!\n");
  }
  else if (h16_c_pos == __float2half(ldexp(1, j + 2)) & h16_c_neg ==
           __float2half(-ldexp(1, j + 2) - ldexp(1, -pout16 + j + 3))) {
    printf("-> Round down\n");
  }
  else {
    printf("-> This test is not working!\n");
  }
#endif


  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               7: Rounding Mode of Aligned Significands
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  /* Assuming FMA size is 3 or more for this case */
  j = 0; // any integer within max min limit definde by exponent
  printheader(outfile, "Feature 6. Rounding Mode for significand alignment");
  host_reset(h_a, h_b, h_c);
  /* neab is computed above, so should work, below works for 1 and 2 */
  h_c[0] = ldexp(1, j);
  h_a[0] = CONVERSION_OP(ldexp(1, -pout + j));
  h_a[1] = h_a[0];
  h_b[0] = CONVERSION_OP(ldexp(1, -neab) + ldexp(1, -neab - 1));
  h_b[1] = h_b[0];
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  h_c_pos = h_c[0];

  h_c[0] = -ldexp(1, j);
  h_a[0] = CONVERSION_OP(-ldexp(1, -pout + j));
  h_a[1] = h_a[0];
  h_b[0] = CONVERSION_OP(ldexp(1, -neab) + ldexp(1, -neab - 1));
  h_b[1] = h_b[0];
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  h_c_neg = h_c[0];

  if (h_c_pos == ldexp(1, j) & h_c_neg == -ldexp(1, j)) {
    printf("-> Round towards zero/truncation\n");
  }

  else if (h_c_pos == (ldexp(1, j) + ldexp(1, -pout + j + 2 - neab)) &
           h_c_neg == -(ldexp(1, j) + ldexp(1, -pout + j + 2 - neab))) {
    printf("-> Round to nearest (even)\n");
  }
  else if (h_c_pos == (ldexp(1, j) + ldexp(1, -pout + j + 2 - neab)) &
           h_c_neg == -(ldexp(1, j))) {
    printf("-> Round up!\n");
  }
  else if (h_c_pos == (ldexp(1, j)) & h_c_neg ==
           -(ldexp(1, j) + ldexp(1, -pout + j + 2 - neab))) {
    printf("-> Round down\n");
  }
  else {
    printf("-> This test is not working!\n");
  }


  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               8: Rounding Mode in Compilation of Multiple BFMAs
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  /* Assuming FMA size is 3 or more for this case */
  j = 0; // any integer within max min limit definde by exponent
  printheader(outfile,
    "Feature 7. Rounding Mode Compilation of Multiple BFMA output");
  host_reset(h_a, h_b, h_c);
  /* NFMA has been computed above, so should work */
  h_c[0] = 1 + ldexp(1, -pout + 1);
  h_a[NFMA] = CONVERSION_OP(ldexp(1, -pout));
  h_b[NFMA] = CONVERSION_OP(ldexp(1, 0) + ldexp(1, -1));
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  h_c_pos = h_c[0];

  host_reset(h_a, h_b, h_c);
  h_c[0] = -1 - ldexp(1, -pout + 1);
  h_a[NFMA] = CONVERSION_OP(-ldexp(1, -pout));
  h_b[NFMA] = CONVERSION_OP(ldexp(1, 0) + ldexp(1, -1));
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  h_c_neg = h_c[0];
  OUT_FORMAT rz_const, rne_const;
  rz_const = 1 + ldexp(1, -pout + 1);
  rne_const = 1 + ldexp(1, -pout + 2);

  if (h_c_pos == rz_const & h_c_neg == -rz_const) {
    printf("-> Round towards zero/truncation\n");
  }
  else if (h_c_pos == rne_const & h_c_neg == -rne_const) {
    printf("-> Round to nearest (even)\n");
  }
  else if (h_c_pos == rne_const & h_c_neg == -rz_const) {
    printf("-> Round up!\n");
  }
  else if (h_c_pos == rz_const & h_c_neg == -rne_const) {
    printf("-> Round down\n");
  }
  else {
    printf("-> This test is not working!\n");
  }

  /*------------------------------------------------------------
   * ------------------------------------------------------------
   *
   *               9: Rounding Mode in Alignment of Significands
   *
   * ------------------------------------------------------------
   --------------------------------------------------------------*/
  /* Assuming FMA size is 3 or more for this case */
  j = 0; // any integer within max min limit definde by exponent
  printheader(outfile, "Feature 9. Rounding Mode in Significands Alignment");
  host_reset(h_a, h_b, h_c);
  /* NFMA has been computed above, so should work */
  h_c[0] = pow(2, j);
  h_a[0] = CONVERSION_OP(ldexp(1, -pout + j));
  h_a[1] = CONVERSION_OP(ldexp(1, -pout + j));

  h_b[0] = CONVERSION_OP(ldexp(1, -neab) + ldexp(1, -neab - 1));
  h_b[1] = CONVERSION_OP(ldexp(1, -neab) + ldexp(1, -neab - 1));
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
  h_c_pos = h_c[0];

  host_reset(h_a, h_b, h_c);
  h_c[0] = -pow(2, j);
  h_a[0] = CONVERSION_OP(-ldexp(1, -pout + j));
  h_a[1] = CONVERSION_OP(-ldexp(1, -pout + j));

  h_b[0] = CONVERSION_OP(ldexp(1, -neab) + ldexp(1, -neab - 1));
  h_b[1] = CONVERSION_OP(ldexp(1, -neab) + ldexp(1, -neab - 1));
  wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);

  h_c_neg = h_c[0];

  rz_const = pow(2, j);
  rne_const = pow(2, j) + ldexp(1, -pout + j + 2 - neab);

  if (h_c_pos == rz_const & h_c_neg == -rz_const) {
    printf("-> Round towards zero/truncation\n");
  }
  else if (h_c_pos == rne_const & h_c_neg == -rne_const) {
    printf("-> Round to nearest (even)\n");
  }
  else if (h_c_pos == rne_const & h_c_neg == -rz_const) {
    printf("-> Round up!\n");
  }
  else if (h_c_pos == rz_const & h_c_neg == -rne_const) {
    printf("-> Round down\n");
  }
  else {
    printf("-> This test is not working!\n");
  }


  /* Free dynamically allocated memory. */
  free(h_a);
  free(h_b);
  free(h16_c);
  cudaFree(d16_a);
  cudaFree(d16_b);
  cudaFree(d16_c);
  cudaFree(d_c);
  free(h_c);
}
