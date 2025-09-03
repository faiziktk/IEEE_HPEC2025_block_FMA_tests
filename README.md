HPEC-25 Paper Simulation and Verification Files
Title: **Generalized Methodology for Determining Numerical Features of Hardware Floating-Point Matrix Multipliers: Part I**
Author: **Faizan A Khattak**, **Mantas Mikaitis**
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

**CUDA C++ File**
1. **BF16**.cu is **CUDA** program file for brain float 16 tensor core numerical feature determination where test vectors are obtained from the above paper.
2. **FP16**.cu is **CUDA** program file for brain float 16 tensor core numerical feature determination where test vectors are obtained from the above paper.
3. **TF32**.cu is **CUDA** program file for brain float 16 tensor core numerical feature determination where test vectors are obtained from the above paper.
4. These files can be run as they are on windows machine, for linux, may be some other header file have been to included.


**MATLAB Files**
These files require CPFloat library to be installed in Matlab
Precision bits for input and output include both mantissa and implicit bit
**A100InnPrdModel.m** is the model where 
   
