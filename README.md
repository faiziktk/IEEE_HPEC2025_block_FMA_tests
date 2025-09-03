HPEC-25 Paper Simulation and Verification Files
Title: **Generalized Methodology for Determining Numerical Features of Hardware Floating-Point Matrix Multipliers: Part I**
Author: **Faizan A Khattak**, **Mantas Mikaitis**
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

**CUDA C++ File**
1. **BF16**.cu is **CUDA** program file for brain float 16 tensor core numerical feature determination where test vectors are obtained from the above paper.
2. **FP16**.cu is **CUDA** program file for half precision binary 16 tensor core numerical feature determination where test vectors are obtained from the above paper.
3. **TF32**.cu is **CUDA** program file for tensor float 32 tensor core numerical feature determination where test vectors are obtained from the above paper.
4. These files can be run as they are on windows machine, for linux, may be some other header file have been to included.


**MATLAB Files**
These files require CPFloat library to be installed in Matlab\\
Precision bits for input and output include both mantissa and implicit bit
**A100InnPrdModel.m** is the model where alignment and normalisation rounding mode, extra alignment bits (neab) and FMA size can be set to model different model for BFMA
<img width="881" height="140" alt="image" src="https://github.com/user-attachments/assets/5e1ab432-ff82-467f-af54-d8d85dced272" />

**HPEC_Test_File.m** applies the test vectors of the above paper to **A100InnPrdModel**. A sample output is shown below

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

HPEC-25: Tensor Core Matrix Multipliers Numerical Feature Testing
Authors: Faizan A. Khattak, Mantas Mikaitis

=================================================================

Numerical Feature 1: Extra Alignment Bit Test
Extra alignment bits detected = 1

===============================================================

Numerical Feature 2: Rounding/Truncation of Aligned Significands
Detected Alignment Rounding Mode: RD (Round Down)

===============================================================

Numerical Feature 3: Final Output Rounding Mode in fp32 Output Mode
Detected Final Output Rounding Mode: RNE (Round to Nearest, ties to Even)

===============================================================

Numerical Feature 4: Final Output Rounding Mode in fp16/bf16 Output Mode
Detected Final Output Rounding Mode: RNE (Round to Nearest, ties to Even)

===============================================================

Numerical Feature 5: Number of Extra Carry Bits in Accumulator for fp16/bf16 Input
Detected Number of Extra Carry Bits = 3

===============================================================

Numerical Feature 6: FMA Size for fp16/bf16 Input
Detected FMA Size = 8

===============================================================

Numerical Feature 7: Number of Extra Carry Bits in Accumulator for TF32 Input
Detected Number of Extra Carry Bits = 2

===============================================================

Numerical Feature 8: FMA Size for TF32 Input
Detected FMA Size = 4

===============================================================

Numerical Feature 9: Rounding Mode in Compilation of Multiple BFMAs Results
RNE (Round to nearest even)
Multiple Block FMA compilation for two BFMAs may reflect internal alignment rounding mode

===============================================================

Numerical Feature 10: Normalisation Pattern Within a BFMA
Delayed/Late normalization

**===END===END===END===END===END===END===END===END===END===END===END**
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

For simulating different modes, change AlignRoundMode, NormRoundMode, extra alignment bits via neab or the FMA size.

