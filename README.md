


**CUDA C++ File**
1. [**BF16**.cu](BF16.cu) is **CUDA** program file for brain float 16 tensor core numerical feature determination where test vectors are obtained from the above paper.
2. [**FP16**.cu](FP16.cu) is **CUDA** program file for half precision (binary 16) tensor core numerical feature determination where test vectors are obtained from the above paper.
3. [**TF32**.cu](TF32.cu) is **CUDA** program file for tensor float 32 tensor core numerical feature determination where test vectors are obtained from the above paper.
4. These files can be run as they are on windows machine, for linux, may be some other header file have been to included.<br>
   **Note**: These tests are generalized but partially based on the previous work of M. Fasi et. al. whose work can be found at [link](https://github.com/north-numerical-computing/tensor-cores-numerical-behavior)<br>  

A sample output of the CUDA file for FP16 is shown below:<br>
<img width="597" height="662" alt="image" src="https://github.com/user-attachments/assets/743c88c7-113b-42cd-9a96-85dcc9d5864a" /><br>

---------------------------------------------------------------------------------------------------------------------------------------<br>

**MATLAB Files**<br>
These files require CPFloat library (can be found at [link](https://github.com/north-numerical-computing/cpfloat)) to be installed in Matlab.<br>
Precision bits for input and output implicitly consider the implicit bit
[**A100InnPrdModel**.m](A100InnPrdModel.m) is the model where alignment and normalisation rounding mode, extra alignment bits (neab) and FMA size can be set to model different model for BFMA or inner product. See a sample below where these parameters can be varied.<br>
<img width="881" height="140" alt="image" src="https://github.com/user-attachments/assets/5e1ab432-ff82-467f-af54-d8d85dced272" /><br>

[**HPEC_Test_File**.m](HPEC_Test_File.m) applies the test vectors of the above paper to [**A100InnPrdModel**.m](A100InnPrdModel.m).<br> 
A sample output is shown below<br>


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

Numerical Feature 10: Normalisation Pattern Within a BFMA<br>
Delayed/Late normalization<br>

--------------------------------------------------------------------------------------------------------------------------
For simulating different BFMA models, change AlignRoundMode, NormRoundMode, extra alignment bits via neab, and the FMA size.

References
--
HPEC-25 Paper Simulation and Verification Files<br>
Title: **Generalized Methodology for Determining Numerical Features of Hardware Floating-Point Matrix Multipliers: Part I**<br>
Author: **Faizan A Khattak**, **Mantas Mikaitis**<br>
Conference: **HPEC** 2025.
