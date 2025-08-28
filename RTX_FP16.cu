/*
 HPEC-25_paper CUDA code for RTX 3060 and Ada Lovelac 1000
FP16 Input
 */

#include <assert.h>
#include <bitset>
 //#include <unistd.h>
#include <cstdint>
#include <chrono>
#include<cmath>
#include <iostream>
#include <mma.h>
#include <iomanip>

//#include <iostream>
#include <bitset>
//#include <cstdint>
using namespace nvcuda;

#define M 16
#define N 16
#define K 16


/*******************
 * Debug functions *
 *******************/
 /* Print the elements of the m x n matrix A. The elements are assumed to be
    stored by columns if `bycols` is `true` and by rows if `bycols` is false. */
template <typename floattype>
void print_matrix(half* a,
    size_t m, size_t n,
    bool bycols) {
    int i, j;
    if (bycols) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++)
                std::cout << __half2float(a[j * n + i]) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    else {
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++)
                std::cout << __half2float(a[i * m + j]) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


/****************************************************
 * Memory management and wmma::mma_sync() interface *
 ****************************************************/

 /* Set the entries of host arrays to zero. */
template <typename returntype>
void host_reset(half* a, half* b, returntype* c) {
    memset(a, 0, 16 * 16 * sizeof(half));
    memset(b, 0, 16 * 16 * sizeof(half));
    memset(c, 0, 16 * 16 * sizeof(returntype));
}

/* Compute C += A*B, where A, B, and C are 16x16x16 matrices.
   The matrix C is initialized to 0 when `init` is true. */
template <typename returntype>
__global__ void wmma_ker(half* a, half* b, returntype* c, bool init) {

 // ----- Declare fragments ----
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, returntype> c_fragment;

    // Load input matrices and initialize output (if required).
    wmma::load_matrix_sync(a_fragment, a, M);
    wmma::load_matrix_sync(b_fragment, b, M);
    if (init)
        wmma::fill_fragment(c_fragment, 0.0f);
    else
        wmma::load_matrix_sync(c_fragment, c, M, wmma::mem_col_major);

    // Multiply
    wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

    // Store the output
    wmma::store_matrix_sync(c, c_fragment, M, wmma::mem_col_major);
}

/* Copy data from host to device, perform the operation, and copy result back to
   host. */
template <typename returntype>
void wmma_init_run(half* h_a, half* h_b, returntype* h_c,
    half* d_a, half* d_b, returntype* d_c,
    bool init) {

    // Copy input from host to device.
    cudaMemcpy(d_a, h_a, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);
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
/*-------------------------------------------------------------------------

 *  For Dynamic Parameter Calculation from input parameters


------------------------------------------------------------------------*/

/* 
* The idea is to return everything in 32 bit precision, and then cast it to half
other format if required
*/
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
    out.minsubnormal = ldexp(1, -(out.bias + p - 1));
    out.aboveone = nextafterf(1., 2.),    // smallest float larger than 1.0
    out.belowtwo = 2. - ldexp(1., -p);
    out.belowone = (1. - ldexp(1, -11)); // only this is needed
    out.gapbelowone = 1. - out.belowone;
   

    return out;
}

//=====================================================================================
//  PRINT IEEE754 in bits 
//=====================================================================================
void printIEEE754(float f) {
    // Reinterpret float bits as 32-bit unsigned integer
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);

    // Print binary with spacing: 1 | 8 | 23 (sign | exponent | fraction)
    std::bitset<32> b(bits);
    std::cout << "Float: " << f << "\n";
    std::cout << "IEEE 754 (binary): ";
    std::cout << b[31] << " | ";                   // Sign bit
    for (int i = 30; i >= 23; --i)                 // Exponent bits
        std::cout << b[i];
    std::cout << " | ";
    for (int i = 22; i >= 0; --i)                  // Fraction bits
        std::cout << b[i];
    std::cout << std::endl;
}

//=====================================================================================
//  PRINT IEEE754 Hal Precision in bits 
//=====================================================================================

void printHalfIEEE(__half h) {
    // Get raw 16-bit binary form of __half
    uint16_t raw_bits = *reinterpret_cast<uint16_t*>(&h);
    std::bitset<16> binary(raw_bits);

    // Print full binary representation
    std::cout << "IEEE 754 Half-Precision Representation:\n";
    std::cout << "Binary    : " << binary << "\n";
    std::cout << "Sign      : " << binary[15] << "\n";
    std::cout << "Exponent  : " << binary.to_string().substr(1, 5) << "\n";
    std::cout << "Fraction  : " << binary.to_string().substr(6, 10) << "\n";
}

/***************
 * EXPERIMENTS *
 ***************/
int main(int argc, char** argv) {
    
    //setting up some constants 
    int pin = 11, pout = 24, pout16=11; //-- considering implicit as well
    
    // Declare pointers and allocate memory.
    half* h_a, * h_b, * h16_c, * d16_a, * d16_b, * d16_c;
    float* d_c, * h_c;
    // size 
    h_a = new half[M * N];
    h_b = new half[M * N];
    h_c = new float[M * N];
    h16_c = new half[M * N];

    cudaMalloc(&d16_a, M * N * sizeof(half));
    cudaMalloc(&d16_b, M * N * sizeof(half));
    cudaMalloc(&d16_c, M * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(float));

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
    h_a[0] = __float2half(ldexp(1, (- pout + j)));
    h_b[0] = __float2half(1);
    h_a[1] = __float2half(ldexp(1, (- pout + j) ));
    h_b[1] = __float2half(1);
    h_c[0] = pow(2,j); 
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    //printIEEE754(h_c[0]);
    if (h_c[0] == (pow(2,j) + 2 * ldexp(1, -pout + j)))
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
    h_a[0] = __float2half(ldexp(1, (-pout + j)));
    h_b[0] = __float2half(1);

    h_a[1] = __float2half(ldexp(1, (-pout + j)));
    h_b[1] = __float2half(ldexp(1,-1));
    h_a[2] = __float2half(ldexp(1, (-pout + j)));
    h_b[2] = __float2half(ldexp(1, -1));

    h_c[0] = pow(2,j);
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    //printIEEE754(h_c[0]);
    if (h_c[0] == (pow(2,j) + 2 * ldexp(1, -pout + j)))
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
    int k = 2, necb = 0, NFMA = 0, ecb = 0;;
    float c = 0, prod_r_sum = 0;
    while (k<15)
    {
        host_reset(h_a, h_b, h_c);
        prod_r_sum = 0;
        c = 0;
        // setting c from Algo 1 from HPEC-25 paper
        h_c[0] = 2 - ldexp(1, -pin + 1);
        for (int i = 1; i <= (ceil(log2(k))); i++)
        {
            h_c[0] = h_c[0] + ldexp(1, -pout + i);

        }
        c = h_c[0];
        //setting rs as per Algo 1 from HPEC-25 paper
        for(int i=0;i<(k-1);i++)
        {
            h_a[i] = __float2half(2 - ldexp(1, -pin + 1));
            prod_r_sum= prod_r_sum+__half2float(h_a[i]);
            h_b[i] = __float2half(1);
        }
        h_a[k - 1] = __float2half(ldexp(1, -pout + 1));
        h_b[k - 1] = __float2half(1);
        prod_r_sum = prod_r_sum+__half2float(h_a[k-1]);

        wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
        if (h_c[0] != (c + prod_r_sum))
        {
            NFMA = k - 1;
            necb = log2(ceil(2 * (k - 1) / (2 - ldexp(1, -pin + 1))) - 1);
            if (NFMA > 1)
            {        ecb = 1;
            }
            break;
        }
        k = k + 1;
    }
    printheader(outfile, "Feature 2. Extra carry bits determination fp16/bf16 inputs");// ;
    printf("Number of extra bits detected are: %d\n", necb);
    
    /*------------------------------------------------------------
    * ------------------------------------------------------------
    *
    *               3: FMA Size (Iterative Approach)
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    printheader(outfile, "Feature 3. FMA size for fp16/bf16 inputs");// ;
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
    h_a[0] = __float2half(ldexp(1, -pout + t) + ldexp(1, -pout));
    h_a[1] = __float2half(ldexp(1, -pout + t) + ldexp(1, -pout));
    h_b[0] = __float2half(1);
    h_b[1] = __float2half(1);
    h_c[0] = 1 - (ldexp(1, -pout + t));
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    if (h_c[0]==(1+ldexp(1,-pout+t)+ ldexp(1, -pout + 1)))
    {
        printf("-> Delayed/Late normalisation\n");
    }
    else if (h_c[0]==(1+ldexp(1,-pout+t)))
    { 
        printf("-> Immediate normalisation with RD/RN/TRC/RZ\n");
    } 
    else if (h_c[0]==(1+ldexp(1,-pout+t)+ldexp(1,-pout+2)))
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
    h_a[0] = __float2half(ldexp(1, j));
    h_a[1] = h_a[0];
    h_a[2] = h_a[0];
    h_b[0] = __float2half(1);
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
    
    if (h_c_pos== ldexp(1, j + 2) & h_c_neg==-ldexp(1,j+2))
    {
        printf("-> Round towards zero/truncation\n");

    }

    else if ( h_c_pos == (ldexp(1, j + 2)+ldexp(1,-pout+j+3)) & h_c_neg == -( ldexp(1, j + 2) + ldexp(1, -pout + j + 3) ) )
    {
        printf("-> Round to nearest (even)\n");

    }
    else if (h_c_pos == (ldexp(1, j + 2) + ldexp(1, -pout + j + 3)) & h_c_neg == -(ldexp(1, j + 2)) )
    {
        printf("-> Round up!\n");
    }
    else if (h_c_pos == (ldexp(1, j + 2)) & h_c_neg == -(ldexp(1, j + 2) + ldexp(1, -pout + j + 3)) )
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
    *               6: Rounding Mode In a BFMA for FP16 Output
    *
    * ------------------------------------------------------------
    --------------------------------------------------------------*/
    // assuming FMA size is 3 or more for this case
    j = 0; // any integer within max min limit definde by exponent
    half h16_c_pos = 0, h16_c_neg = 0;
    printheader(outfile, "Feature 6. Rounding Mode for final output in fp16 of a BFMA");// ;
    host_reset(h_a, h_b, h16_c);
    
    h16_c[0] = __float2half(ldexp(1, j) + ldexp(1, -pout16 + j + 1) + ldexp(1, -pout16 + j + 2));
    h_a[0] = __float2half(ldexp(1, j));
    h_a[1] = h_a[0];
    h_a[2] = h_a[0];
    h_b[0] = __float2half(1);
    h_b[1] = h_b[0]; h_b[2] = h_b[1];
    wmma_init_run(h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
    h16_c_pos = h16_c[0];
    // negative axis
    host_reset(h_a, h_b, h16_c);
    h16_c[0] = __float2half(-ldexp(1, j) - ldexp(1, -pout16 + j + 1) - ldexp(1, -pout16 + j + 2));
    h_a[0] = __float2half(-ldexp(1, j));
    h_a[1] = h_a[0];
    h_a[2] = h_a[0];
    h_b[0] = __float2half(1);
    h_b[1] = h_b[0]; h_b[2] = h_b[1];
    wmma_init_run(h_a, h_b, h16_c, d16_a, d16_b, d16_c, false);
    h16_c_neg = h16_c[0];
    

    if (h16_c_pos == __float2half(ldexp(1, j + 2)) & h16_c_neg == __float2half (- ldexp(1, j + 2)))
    {
        printf("-> Round towards zero/truncation\n");

    }

    else if (h16_c_pos == __float2half(ldexp(1, j + 2) + ldexp(1, -pout16 + j + 3)) & h16_c_neg == __float2half(-ldexp(1, j + 2) - ldexp(1, -pout16 + j + 3)))
    {
        printf("-> Round to nearest (even)\n");

    }
    else if (h16_c_pos == __float2half(ldexp(1, j + 2) + ldexp(1, -pout16 + j + 3)) & h16_c_neg == __float2half(-ldexp(1, j + 2)))
    {
        printf("-> Round up!\n");
    }
    else if (h16_c_pos == __float2half(ldexp(1, j + 2)) & h16_c_neg == __float2half(-ldexp(1, j + 2) - ldexp(1, -pout16 + j + 3)))
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
    printheader(outfile, "Feature 7. Rounding Mode for significand alignment");// ;
    host_reset(h_a, h_b, h_c);
    // neab is computed above, so should work, below works for 1 and 2
    h_c[0] = ldexp(1, j);
    h_a[0] = __float2half(ldexp(1, -pout+j));
    h_a[1] = h_a[0];
    h_b[0] = __float2half(ldexp(1,-neab)+ ldexp(1, -neab-1));
    h_b[1] = h_b[0]; 
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    h_c_pos = h_c[0];

    h_c[0] = -ldexp(1, j);
    h_a[0] = __float2half(-ldexp(1, -pout + j));
    h_a[1] = h_a[0];
    h_b[0] = __float2half(ldexp(1, -neab) + ldexp(1, -neab - 1));
    h_b[1] = h_b[0];
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    h_c_neg = h_c[0];

    if (h_c_pos == ldexp(1, j) & h_c_neg == -ldexp(1, j))
    {
        printf("-> Round towards zero/truncation\n");

    }

    else if (h_c_pos == (ldexp(1, j) + ldexp(1, -pout + j + 2-neab)) & h_c_neg == -(ldexp(1, j) + ldexp(1, -pout + j + 2-neab)))
    {
        printf("-> Round to nearest (even)\n");

    }
    else if (h_c_pos == (ldexp(1, j) + ldexp(1, -pout + j + 2-neab)) & h_c_neg == -(ldexp(1, j)))
    {
        printf("-> Round up!\n");
    }
    else if (h_c_pos == (ldexp(1, j)) & h_c_neg == -(ldexp(1, j) + ldexp(1, -pout + j + 2-neab)))
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
    printheader(outfile, "Feature 8. Rounding Mode Compilation of Multiple BFMA output");// ;
    host_reset(h_a, h_b, h_c);
    // NFMA has been computed above, so should work
    h_c[0] = 1+ldexp(1,-pout+1);
    h_a[NFMA] = __float2half(ldexp(1, -pout));
    h_b[NFMA] = __float2half(ldexp(1,0)+ldexp(1,-1));
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    h_c_pos = h_c[0];
    
    host_reset(h_a, h_b, h_c);
    h_c[0] = -1 - ldexp(1, -pout + 1);
    h_a[NFMA] = __float2half(-ldexp(1, -pout));
    h_b[NFMA] = __float2half(ldexp(1, 0) + ldexp(1, -1));
    wmma_init_run(h_a, h_b, h_c, d16_a, d16_b, d_c, false);
    h_c_neg = h_c[0];
    float rz_const, rne_const;
    rz_const = 1 + ldexp(1, -pout + 1);
    rne_const = 1 + ldexp(1, -pout + 2 );
    


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
    free(h_a);
    free(h_b);
    free(h16_c);
    cudaFree(d16_a);
    cudaFree(d16_b);
    cudaFree(d16_c);
    cudaFree(d_c);
    free(h_c);

    system("pause");  // Press any key to continue...
    return 0;

}
