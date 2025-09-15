Generalized Methodology for Determining Numerical Features of Hardware Floating-Point Matrix Multipliers
--
## Overview

This repository provides code that implements the proof-of-concept of the methodology for determining the numerical features of matrix multipliers on modern GPUs, presented in [1]. The code is a based on the code developed by Fasi et al. [2] available at [north-numerical-computing/tensor-cores-numerical-behavior](https://github.com/north-numerical-computing/tensor-cores-numerical-behavior).

The software can perform analysis of the following features of matrix multiplication:

* Support for subnormal numbers
* Presence of extra bits for significand alignment in multi-term addition
* Availability of extra carry bits
* Normalization patterns in multi-term floating-point addition
* Supported rounding modes
* Effective FMA size (i.e., number of terms accumulated before a single normalization)

## Related CUDA Files

[nvidia-tests.cu](CUDA/nvidia-tests.cu): CUDA code that applies the generalized test vectors [1] to test matrix multipliers using the WMMA API, with inputs in binary16/bfloat16/TensorFloat32. The macros at the top of the file allow to select the input format.

## Sample Output of NVIDIA A30 Tensor Cores Numerical Features

A sample output of the CUDA file for FP16 is shown below:<br>

```
$ nvcc -arch=sm_80 nvidia-tests.cu -o nvidia-tests
$./nvidia-tests

+--------------------------------------------------------------+
| Feature 1. Extra Alignment Bits Determination                |
+--------------------------------------------------------------+
-> One bit existence test: Passed
-> Two bits existence test: Failed
+--------------------------------------------------------------+
| Feature 2. Extra carry bits determination fp16/bf16 inputs   |
+--------------------------------------------------------------+
Number of extra bits detected are: 3
+--------------------------------------------------------------+
| Feature 3. FMA size for fp16/bf16 inputs                     |
+--------------------------------------------------------------+
The FMA size is: 8
+--------------------------------------------------------------+
| Feature 4. Normalisation pattern in BFMA                     |
+--------------------------------------------------------------+
-> Delayed/Late normalisation
+--------------------------------------------------------------+
| Feature 5. Rounding Mode for final output in SP of a BFMA    |
+--------------------------------------------------------------+
-> Round towards zero/truncation
+--------------------------------------------------------------+
| Feature 6. Rounding Mode for final output in fp16 of a BFMA  |
+--------------------------------------------------------------+
-> Round to nearest (even)
+--------------------------------------------------------------+
| Feature 6. Rounding Mode for significand alignment           |
+--------------------------------------------------------------+
-> Round towards zero/truncation
+--------------------------------------------------------------+
| Feature 7. Rounding Mode Compilation of Multiple BFMA output |
+--------------------------------------------------------------+
-> Round towards zero/truncation
+--------------------------------------------------------------+
| Feature 9. Rounding Mode in Significands Alignment           |
+--------------------------------------------------------------+
-> Round towards zero/truncation
```

## MATLAB Block Fused Multiply Accumulate Model (BFMA) for Modeling different models of tensor cores

[A100InnPrdModel.m](MATLAB/A100InnPrdModel.m) is the model where alignment and normalisation rounding mode, extra alignment bits (neab) and FMA size can be set to model different model for BFMA or inner product. See a sample below where these parameters can be varied.<br>
<img width="881" height="140" alt="image" src="https://github.com/user-attachments/assets/5e1ab432-ff82-467f-af54-d8d85dced272" /><br>

[HPEC_Test_File.m](MATLAB/HPEC_Test_File.m) applies the test vectors in [1] to [A100InnPrdModel.m](MATLAB/A100InnPrdModel.m).<br> 
Note These files require CPFloat library (can be found at [link](https://github.com/north-numerical-computing/cpfloat)) to be installed in Matlab.<br>
Precision bits for input and output implicitly consider the implicit bit

## Sample Output for Matlab Based BFMA Model
Runing [HPEC_Test_File](MATLAB/HPEC_Test_File.m) would output following numerical features of the model simulated by the user by setting the parameter mentioned above.<br>
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

## References

[1] Faizan A Khattak and Mantas Mikaitis, [Generalized Methodology for Determining Numerical Features of Hardware Floating-Point Matrix Multipliers: Part I](https://eprints.whiterose.ac.uk/id/eprint/231310/). Accepted for 29th Annual IEEE High Performance Extreme Computing. Sep. 2025.<br>

[2] Massimiliano Fasi, Nicholas J. Higham, Mantas Mikaitis, and Srikara Pranesh, [Numerical Behavior of the NVIDIA Tensor Cores](https://peerj.com/articles/cs-330/). PeerJ Computer Science, 7:e330. Feb. 2021.
