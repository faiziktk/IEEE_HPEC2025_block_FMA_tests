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
These files require CPFloat library to be installed in Matlab
Precision bits for input and output include both mantissa and implicit bit
**A100InnPrdModel.m** is the model where alignment and normalisation rounding mode, extra alignment bits (neab) and FMA size can be set to model different model for BFMA
<img width="1038" height="140" alt="image" src="https://github.com/user-attachments/assets/4e6dff24-d130-43c1-a7ca-70c6bc677d6a" />
   
