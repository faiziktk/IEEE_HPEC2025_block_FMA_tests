/*
HPEC-25 Paper CUDA code for TF32 input 
 */

#include <assert.h>
#include<cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <mma.h>
#include <iomanip>
#include <stdio.h>
#include <string.h>

using namespace nvcuda;

/*******************
 * Debug functions *
 *******************/
 /* Print the elements of the m x n matrix A. The elements are assumed to be
    stored by columns if `bycols` is `true` and by rows if `bycols` is false. */
template <typename floattype>
void print_matrix(float* a,
    size_t m, size_t n,
    bool bycols) {
    int i, j;
    if (bycols) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++)
                std::cout << a[j * n + i] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    else {
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++)
                std::cout << a[i * m + j] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


/****************************************************
 * Memory management and wmma::mma_sync() interface *
 ****************************************************/

 /* Set the entries of host arrays to zero. */
void host_reset(float* a, float* b, float* c) {
    memset(a, 0, 16 * 16 * sizeof(float));
    memset(b, 0, 16 * 16 * sizeof(float));
    memset(c, 0, 16 * 16 * sizeof(float));
}

/* Compute C += A*B, where A, B, and C are 16x16x16 matrices.
   The matrix C is initialized to 0 when `init` is true. */
__global__ void wmma_ker(float* a, float* b, float* c, bool init) {

    // Declare fragments.
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
        wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
        wmma::col_major> b_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_fragment;

    // Load input matrices and initialize output (if required).
    wmma::load_matrix_sync(a_fragment, a, 8);
    wmma::load_matrix_sync(b_fragment, b, 8);
    if (init)
        wmma::fill_fragment(c_fragment, 0.0f);
    else
        wmma::load_matrix_sync(c_fragment, c, 16, wmma::mem_col_major);

    // Multiply
    wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

    // Store the output
    wmma::store_matrix_sync(c, c_fragment, 16, wmma::mem_col_major);
}

/* Copy data from host to device, perform the operation, and copy result back to
   host. */
void wmma_init_run(float* h_a, float* h_b, float* h_c,
    float* d_a, float* d_b, float* d_c,
    bool init) {

    // Copy input from host to device.
    cudaMemcpy(d_a, h_a, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication.
    wmma_ker << <1, 32 >> > (d_a, d_b, d_c, init);

    // Copy result from device to host.
    cudaMemcpy(h_c, d_c, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost);
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
void printitem(FILE* outfile, const char* string) {
    fprintf(outfile, "  | %-49s", string);
}

void printpass(FILE* outfile, bool status) {
    if (status)
        fprintf(outfile, " [PASS] |\n");
    else
        fprintf(outfile, " [FAIL] |\n");
}
void printfooter(FILE* outfile) {
    fprintf(outfile,
        "  +----------------------------------------------------------+\n\n");
}

// return type structure defined for multiple stuff
typedef struct {
    int bias;
    float minsubnormal;
    float minposnormal;
    float belowone;
    float gapbelowone;
    float aboveone; 
    float belowtwo;
} ParaRet;
// Just a function
ParaRet CompParaTstVec(int e, int p, const char* str) {
    ParaRet out;
    int bias;
    out.bias = (1 << (e - 1)) - 1;
    out.minposnormal = ldexp(1, -out.bias + 1);
    out.minsubnormal = ldexp(1,-(out.bias+p-1));
    out.aboveone = nextafterf(1., 2.),    // smallest float larger than 1.0
    out.belowtwo = 2. - ldexp(1., -p);
    if (strcmp(str, "TF32")==1 | strcmp(str, "FP32") == 1)
    {
        out.belowone = nextafter(1., 0.);
        out.gapbelowone = 1. - out.belowone;
        

    }
    else if (strcmp(str, "FP16") == 0) 
    {
        //nothign for time being
    }
    else
    { 
    //nothing
    }
    
    
    return out;
}



/***************
 * EXPERIMENTS *
 ***************/
int main(int argc, char** argv) {

    // Declare pointers and allocate memory.
    float* h_a, * h_b, * h_c, * d_a, * d_b, * d_c;
    int pin=11, pout = 24; // input tf32 and output sp


    h_a = new float[16 * 16];
    h_b = new float[16 * 16];
    h_c = new float[16 * 16];

    cudaMalloc(&d_a, 16 * 16 * sizeof(float));
    cudaMalloc(&d_b, 16 * 16 * sizeof(float));
    cudaMalloc(&d_c, 16 * 16 * sizeof(float));

    FILE* outfile = stdout;
    bool pass;



    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               1: Extra Alignment Bit test: (one and two bits)
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    int j = 0, eab = 0, neab = 0;  // j can be made any integer dependent upon exponent of input/output 
    printheader(outfile, "Feature 1. Extra Alignment Bits Determination");// ;
    host_reset(h_a, h_b, h_c);
    //single bit test existence
    h_a[0] = (ldexp(1, (-pout + j)));
    h_b[0] = (1);
    h_a[1] = (ldexp(1, (-pout + j)));
    h_b[1] = (1);
    h_c[0] = pow(2, j);
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    //printIEEE754(h_c[0]);
    if (h_c[0] == (pow(2, j) + 2 * ldexp(1, -pout + j)))
    {
        printf("-> One bit existence test: Passed\n");
        eab = 1; /*extra alignment bit exist*/
        neab = 1;
    }
    else
    {
        printf("-> One bit existence test: Failed\n");
    }

    //two bits test existence
    host_reset(h_a, h_b, h_c);
    h_a[0] = (ldexp(1, (-pout + j)));
    h_b[0] = (1);

    h_a[1] = (ldexp(1, (-pout + j)));
    h_b[1] = (ldexp(1, -1));
    h_a[2] = (ldexp(1, (-pout + j)));
    h_b[2] = (ldexp(1, -1));

    h_c[0] = pow(2, j);
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    //printIEEE754(h_c[0]);
    if (h_c[0] == (pow(2, j) + 2 * ldexp(1, -pout + j)))
    {
        printf("-> Two bits existence test: Passed\n");
        neab = 2;
        eab = 1; /*extra alignment bit exist*/
    }
    else
    {
        printf("->Two bits existence test: Failed\n");
    }


    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               2: Extra Carry Bits (Iterative Approach)
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    /*Assuming the FMA size is less than permitted max shared dimension of tc*/
    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               2: Extra Carry Bits (Iterative Approach)
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    /*Assuming the FMA size is less than permitted max shared dimension of tc*/
    int k = 2, necb = 0, NFMA = 0, ecb = 0;;
    float c = 0, d_pos = 0, d_neg = 0;
    while (k < 15)
    {
        // positive axis
        host_reset(h_a, h_b, h_c);
        h_c[0] = 1 + ldexp(1, -pout + 1);
        h_a[0] = (1);
        h_b[0] = (1);
        h_b[k - 1] = h_b[0];
        h_a[k - 1] = (ldexp(1, -pout + 1));
        wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
        d_pos = h_c[0];
        // negative axis
        host_reset(h_a, h_b, h_c);
        h_c[0] = -1 - ldexp(1, -pout + 1);
        h_a[0] = (-1);
        h_b[0] = (1);
        h_b[k - 1] = h_b[0];
        h_a[k - 1] = (-ldexp(1, -pout + 1));
        wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
        d_neg = h_c[0];
        if (d_pos != (2 + ldexp(1, -pout + 2)) && abs(d_neg) != (2 + ldexp(1, -pout + 2)))
        {
            NFMA = k - 1;
            ecb = 1;
            
            break;
        }
        necb = floor(log2(k * (2 - ldexp(1, -pin + 1))));
        k = k + 1;
    }
    printheader(outfile, "Feature 2. Extra carry bits determination TF32 inputs");// ;
    printf("Number of extra bits detected are: %d\n", necb);
    
    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               3: FMA Size (Iterative Approach)
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    printheader(outfile, "Feature 3. FMA size for TF32 inputs");// ;
    printf("The FMA size is: %d\n", NFMA);


    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               4: Normalisation Pattern Within A Block FMA
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    // eab=1,ecb=1; in this case, 
    printheader(outfile, "Feature 4. Normalisation pattern in BFMA");// ;
    int t = 3; // for example, from HPEC-25 paper
    host_reset(h_a, h_b, h_c);
    h_a[0] = (ldexp(1, -pout + t) + ldexp(1, -pout));
    h_a[1] = (ldexp(1, -pout + t) + ldexp(1, -pout));
    h_b[0] = (1);
    h_b[1] = (1);
    h_c[0] = 1 - (ldexp(1, -pout + t));
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    if (h_c[0] == (1 + ldexp(1, -pout + t) + ldexp(1, -pout + 1)))
    {
        printf("-> Delayed/Late normalisation\n");
    }
    else if (h_c[0] == (1 + ldexp(1, -pout + t)))
    {
        printf("-> Immediate normalisation with RD/RN/TRC/RZ\n");
    }
    else if (h_c[0] == (1 + ldexp(1, -pout + t) + ldexp(1, -pout + 2)))
    {
        printf("-> Immediate normalisation with RU\n");
    }
    else
    {
        printf("-> This test is not working!\n");
    }


    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               5: Rounding Mode In a BFMA
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    // assuming FMA size is 3 or more for this case
    j = 0; // any integer within max min limit definde by exponent
    float h_c_pos = 0, h_c_neg = 0, h_c_temp = 0;
    printheader(outfile, "Feature 5. Rounding Mode for final output in SP of a BFMA");// ;
    h_c[0] = ldexp(1, j) + ldexp(1, -pout + j + 1) + ldexp(1, -pout + j + 2);
    h_c_temp = h_c[0];
    h_a[0] = (ldexp(1, j));
    h_a[1] = h_a[0];
    h_a[2] = h_a[0];
    h_b[0] = (1);
    h_b[1] = h_b[0]; h_b[2] = h_b[1];
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    h_c_pos = h_c[0];
    // negative axis
    h_c[0] = -h_c_temp;
    h_a[0] = -h_a[0];
    h_a[1] = h_a[0];
    h_a[2] = h_a[0];
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    h_c_neg = h_c[0];

    if (h_c_pos == ldexp(1, j + 2) & h_c_neg == -ldexp(1, j + 2))
    {
        printf("-> Round towards zero/truncation\n");

    }

    else if (h_c_pos == (ldexp(1, j + 2) + ldexp(1, -pout + j + 3)) & h_c_neg == -(ldexp(1, j + 2) + ldexp(1, -pout + j + 3)))
    {
        printf("-> Round to nearest (even)\n");

    }
    else if (h_c_pos == (ldexp(1, j + 2) + ldexp(1, -pout + j + 3)) & h_c_neg == -(ldexp(1, j + 2)))
    {
        printf("-> Round up!\n");
    }
    else if (h_c_pos == (ldexp(1, j + 2)) & h_c_neg == -(ldexp(1, j + 2) + ldexp(1, -pout + j + 3)))
    {
        printf("-> Round down\n");
    }
    else
    {
        printf("-> This test is not working!\n");
    }





    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               7: Rounding Mode of Aligned Significands
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    // assuming FMA size is 3 or more for this case
    j = 0; // any integer within max min limit definde by exponent
    printheader(outfile, "Feature 6. Rounding Mode for significand alignment");// ;
    host_reset(h_a, h_b, h_c);
    // neab is computed above, so should work, below works for 1 and 2
    h_c[0] = ldexp(1, j);
    h_a[0] = (ldexp(1, -pout + j));
    h_a[1] = h_a[0];
    h_b[0] = (ldexp(1, -neab) + ldexp(1, -neab - 1));
    h_b[1] = h_b[0];
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    h_c_pos = h_c[0];

    h_c[0] = -ldexp(1, j);
    h_a[0] = (-ldexp(1, -pout + j));
    h_a[1] = h_a[0];
    h_b[0] = (ldexp(1, -neab) + ldexp(1, -neab - 1));
    h_b[1] = h_b[0];
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    h_c_neg = h_c[0];

    if (h_c_pos == ldexp(1, j) & h_c_neg == -ldexp(1, j))
    {
        printf("-> Round towards zero/truncation\n");

    }

    else if (h_c_pos == (ldexp(1, j) + ldexp(1, -pout + j + 2 - neab)) & h_c_neg == -(ldexp(1, j) + ldexp(1, -pout + j + 2 - neab)))
    {
        printf("-> Round to nearest (even)\n");

    }
    else if (h_c_pos == (ldexp(1, j) + ldexp(1, -pout + j + 2 - neab)) & h_c_neg == -(ldexp(1, j)))
    {
        printf("-> Round up!\n");
    }
    else if (h_c_pos == (ldexp(1, j)) & h_c_neg == -(ldexp(1, j) + ldexp(1, -pout + j + 2 - neab)))
    {
        printf("-> Round down\n");
    }
    else
    {
        printf("-> This test is not working!\n");
    }



    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               8: Rounding Mode in Compilation of Multiple BFMAs
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    // assuming FMA size is 3 or more for this case
    j = 0; // any integer within max min limit definde by exponent
    printheader(outfile, "Feature 7. Rounding Mode Compilation of Multiple BFMA output");// ;
    host_reset(h_a, h_b, h_c);
    // NFMA has been computed above, so should work
    h_c[0] = 1 + ldexp(1, -pout + 1);
    h_a[NFMA] = (ldexp(1, -pout));
    h_b[NFMA] = (ldexp(1, 0) + ldexp(1, -1));
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    h_c_pos = h_c[0];

    host_reset(h_a, h_b, h_c);
    h_c[0] = -1 - ldexp(1, -pout + 1);
    h_a[NFMA] = (-ldexp(1, -pout));
    h_b[NFMA] = (ldexp(1, 0) + ldexp(1, -1));
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    h_c_neg = h_c[0];
    float rz_const, rne_const;
    rz_const = 1 + ldexp(1, -pout + 1);
    rne_const = 1 + ldexp(1, -pout + 2);



    if (h_c_pos == rz_const & h_c_neg == -rz_const)
    {
        printf("-> Round towards zero/truncation\n");

    }

    else if (h_c_pos == rne_const & h_c_neg == -rne_const)
    {
        printf("-> Round to nearest (even)\n");

    }
    else if (h_c_pos == rne_const & h_c_neg == -rz_const)
    {
        printf("-> Round up!\n");
    }
    else if (h_c_pos == rz_const & h_c_neg == -rne_const)
    {
        printf("-> Round down\n");
    }
    else
    {
        printf("-> This test is not working!\n");
    }


    

    /*------------------------------------------------------------
        * ------------------------------------------------------------
        *
        *               9: Rounding Mode in Alignment of Significands
        *
        * ------------------------------------------------------------
        --------------------------------------------------------------*/
        // assuming FMA size is 3 or more for this case
    j = 0; // any integer within max min limit definde by exponent
    printheader(outfile, "Feature 8. Rounding Mode in Significands Alignment");// ;
    host_reset(h_a, h_b, h_c);
    // NFMA has been computed above, so should work
    h_c[0] = pow(2, j);
    h_a[0] = (ldexp(1, -pout + j));
    h_a[1] = (ldexp(1, -pout + j));

    h_b[0] = (ldexp(1, -neab) + ldexp(1, -neab - 1));
    h_b[1] = (ldexp(1, -neab) + ldexp(1, -neab - 1));
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);
    h_c_pos = h_c[0];

    host_reset(h_a, h_b, h_c);
    h_c[0] = -pow(2, j);
    h_a[0] = (-ldexp(1, -pout + j));
    h_a[1] = (-ldexp(1, -pout + j));

    h_b[0] = (ldexp(1, -neab) + ldexp(1, -neab - 1));
    h_b[1] = (ldexp(1, -neab) + ldexp(1, -neab - 1));
    wmma_init_run(h_a, h_b, h_c, d_a, d_b, d_c, false);

    h_c_neg = h_c[0];

    rz_const = pow(2, j);
    rne_const = pow(2, j) + ldexp(1, -pout + j + 2 - neab);



    if (h_c_pos == rz_const & h_c_neg == -rz_const)
    {
        printf("-> Round towards zero/truncation\n");

    }

    else if (h_c_pos == rne_const & h_c_neg == -rne_const)
    {
        printf("-> Round to nearest (even)\n");

    }
    else if (h_c_pos == rne_const & h_c_neg == -rz_const)
    {
        printf("-> Round up!\n");
    }
    else if (h_c_pos == rz_const & h_c_neg == -rne_const)
    {
        printf("-> Round down\n");
    }
    else
    {
        printf("-> This test is not working!\n");
    }




    // Free dynamically allocated memory.
    //  free(h_a);
    //  free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
