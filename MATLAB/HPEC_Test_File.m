clear all;
close all;
clc;

% Example by Faizan A. Khattak, 18/07/2025
% Test script for A100/Ada1000 Tensor Core Dot Product Model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input and output format configuration
% Note: the default rounding mode in CPFloat is RN-TE.
inoptions.format  = 'binary16';   % Input format
outoptions.format = 'binary32';   % Output format

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Numerical feature test: extra alignment bits
% This script evaluates the number of additional alignment bits required.
% Only two bits are tested here. For extended results, refer to:
%   HPEC'25 paper: F. A. Khattak, M. Mikaitis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pout = 24;              % Output precision (single-precision mantissa bits)
j    = 0;               % Must be a non-negative integer

% --- Test case 1: single extra alignment bit
[a, opts_in] = cpfloat([2^(-pout+j), 2^(-pout+j), 0, 0, 0, 0, 0, 0], inoptions);
[b, opts_in] = cpfloat([1, 1, 1, 1, 1, 1, 1, 1], inoptions);
[c, opts_out] = cpfloat(2^j, outoptions);

[d, dbits, exp, sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
neab = (d == (a*b' + c));   % neab = 1 if match, otherwise 0

% --- Test case 2: two extra alignment bits
[a, opts_in] = cpfloat([2^(-pout+j), 2^(-pout+j), 2^(-pout+j), 0, 0, 0, 0, 0], inoptions);
[b, opts_in] = cpfloat([2^-1, 2^-1, 1, 1, 1, 1, 1, 1], inoptions);

[d, dbits, exp, sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);

if (d == (a*b' + c))
    neab = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
disp('HPEC-25: Tensor Core Matrix Multipliers Numerical Feature Testing');
disp('Authors: Faizan A. Khattak, Mantas Mikaitis');
disp('=================================================================');
disp('Numerical Feature 1: Extra Alignment Bit Test');
disp(['Extra alignment bits detected = ', num2str(neab)]);

if neab == 2
    disp('Note: Additional extra alignment bits may exist beyond this test.');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Feature 2: Rounding/Truncation at Alignment of Significands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assumption: the model uses one extra alignment bit
% Configuration: input format = bfloat16, output format = binary32
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('===============================================================');
disp('Numerical Feature 2: Rounding/Truncation of Aligned Significands');

if neab < 2
    % Parameters
    pout = 24;   % Single-precision mantissa bits
    pin  = 11;   % bfloat16 mantissa bits (including implicit bit)
    j    = 0;    % Must be a non-negative integer

    % Construct input and output test vectors
    [a, opts_in]  = cpfloat([2^(-pout+j), 2^(-pout+j), 0, 0, 0, 0, 0, 0], inoptions);
    [b, opts_in]  = cpfloat([sum(2.^([-neab, -neab-1])), sum(2.^([-neab, -neab-1])), 1, 1, 1, 1, 1, 1], inoptions);
    [c, opts_out] = cpfloat(2^j, outoptions);

    % Perform inner product operations
    [dp, dbits, exp, signp] = A100InnPrdModel( a,  b,  c, inoptions, outoptions);
    [dn, dbits, exp, signn] = A100InnPrdModel(-a,  b, -c, inoptions, outoptions);

    % Expected constants for different rounding modes
    rzconst  = 2^j;
    rneconst = 2^j + 2^(-pout + j - neab + 2);

    % Rounding mode detection
    if (dp == rneconst && abs(dn) == rneconst)
        disp('Detected Alignment Rounding Mode: RNE (Round to Nearest, ties to Even)');
    elseif (dp == rzconst && abs(dn) == rzconst)
        disp('Detected Alignment Rounding Mode: Truncation (Round toward Zero)');
    elseif (dp == rneconst && abs(dn) == rzconst)
        disp('Detected Alignment Rounding Mode: RU (Round Up)');
    elseif (dp == rzconst && abs(dn) == rneconst)
        disp('Detected Alignment Rounding Mode: RD (Round Down)');
    else
        disp('Rounding mode could not be determined with this test.');
    end
else
    disp('Alignment rounding mode test is supported only for neab = 0 and neab = 1.');
    disp('Tests for higher values of neab will be presented in Part II.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Feature 3: Final Output Rounding Mode
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assumption: No intermediate rounding occurs due to the block-FMA property.
% This test identifies the rounding mode applied at the final fp32 output stage.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('===============================================================');
disp('Numerical Feature 3: Final Output Rounding Mode in fp32 Output Mode');

% Parameters
pout = 24;   % Single-precision mantissa bits
pin  = 11;   % bfloat16 mantissa bits (including implicit bit)
j    = 0;    % Must be non-negative integer

% Construct input and output test vectors
[a, opts_in]  = cpfloat([2^j, 2^j, 2^j, 0, 0, 0, 0, 0], inoptions);
[b, opts_in]  = cpfloat([1, 1, 1, 1, 1, 1, 1, 1], inoptions);
[c, opts_out] = cpfloat(2^j + 2^(-pout+j+1) + 2^(-pout+j+2), outoptions);

% Perform inner product operations
[dp, dbits, exp, sign] = A100InnPrdModel( a,  b,  c, inoptions, outoptions);
[dn, dbits, exp, sign] = A100InnPrdModel(-a,  b, -c, inoptions, outoptions);

% Expected constants for different rounding modes
rzconst = 2^(j+2);
rneconst = rzconst + 2^(-pout + j + 3);

% Rounding mode detection
if (dp == rneconst && abs(dn) == rneconst)
    disp('Detected Final Output Rounding Mode: RNE (Round to Nearest, ties to Even)');
elseif (dp == rzconst && abs(dn) == rzconst)
    disp('Detected Final Output Rounding Mode: Truncation (Round toward Zero)');
elseif (dp == rneconst && abs(dn) == rzconst)
    disp('Detected Final Output Rounding Mode: RU (Round Up)');
elseif (dp == rzconst && abs(dn) == rneconst)
    disp('Detected Final Output Rounding Mode: RD (Round Down)');
else
    disp('Final output rounding mode could not be determined with this test.');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Feature 4: Final Output Rounding Mode (fp16/bf16 Output Mode)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assumption: No intermediate rounding occurs due to the block-FMA property.
% This test determines the rounding mode applied at the final fp16/bf16 
% output stage.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

outoptions.format = 'binary16';

disp('===============================================================');
disp('Numerical Feature 4: Final Output Rounding Mode in fp16/bf16 Output Mode');

% Parameters
pout = 11;   % Mantissa bits for binary16/bfloat16 (including implicit bit)
pin  = 11;
j    = 0;    % Non-negative integer

% Construct input and output test vectors
[a, opts_in]  = cpfloat([0, 0, 0, 2^(-pin+j), 2^(-pin-1+j), 0, 0, 0], inoptions);
[b, opts_in]  = cpfloat([1, 1, 1, 1, 1, 1, 1, 1], inoptions);
[c, opts_out] = cpfloat(2^j, outoptions);

% Perform inner product operations
[dp, dbits, exp, sign] = A100InnPrdModel( a,  b,  c, inoptions, outoptions);
[dn, dbits, exp, sign] = A100InnPrdModel(-a,  b, -c, inoptions, outoptions);

% Expected constants for different rounding modes
rzconst  = 2^j;
rneconst = rzconst + 2^(-pin+1);

% Rounding mode detection
if (dp == rneconst && abs(dn) == rneconst)
    disp('Detected Final Output Rounding Mode: RNE (Round to Nearest, ties to Even)');
elseif (dp == rzconst && abs(dn) == rzconst)
    disp('Detected Final Output Rounding Mode: Truncation (Round toward Zero)');
elseif (dp == rneconst && abs(dn) == rzconst)
    disp('Detected Final Output Rounding Mode: RU (Round Up)');
elseif (dp == rzconst && abs(dn) == rneconst)
    disp('Detected Final Output Rounding Mode: RD (Round Down)');
else
    disp('Final output rounding mode could not be determined with this test.');
end


%%--------------------------------------------------------------------
%% Numerical Feature 5: Extra Carry Bits in the Accumulator (fp16/bf16 Input)
%%--------------------------------------------------------------------
disp('===============================================================');
disp('Numerical Feature 5: Number of Extra Carry Bits in Accumulator for fp16/bf16 Input');

% Input/output format setup
inoptions.format  = 'binary16';
outoptions.format = 'binary32';
inputsize = 16;

% Parameters
pin  = 11;  % Mantissa bits for fp16 input (with implicit bit)
pout = 24;  % Mantissa bits for fp32 output (with implicit bit)
j    = 0;   % Offset parameter for scaling
k    = 2;   % Starting index for HPEC Algorithm 1
necb = 0;   % Number of extra carry bits (to be detected)

while true
    lgk = ceil(log2(k));

    % Construct test input vectors (carry-propagation detection)
    a = zeros(1, inputsize);
    b = ones(1, inputsize);
    c = 2^j + 2^(-pout+1+j);
    a(1) = 2^j;
    a(k) = 2^(-pout+1+j);

    % Tensor core inner product evaluation
    [dp, dbits, exp, sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
    [dn, dbits, exp, sign] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);

    % Detect termination: FMA block size
    if dp ~= (2^(j+1) + 2^(-pout+2+j)) && abs(dn) ~= (2^(j+1) + 2^(-pout+2+j))
        NFMA = k - 1;
        break;
    end

    % Update for extra carry-bit detection
    a = zeros(1, inputsize);
    b = ones(1, inputsize);
    c = 2 - 2^(-pin+1) + sum(2.^(-pout+1 : -pout+lgk));
    a(1:k-1) = 2 - 2^(-pin+1);
    a(k)     = 2^(-pout+1);

    [dp, dbits, exp, sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);

    % If full propagation occurs, update extra carry bits
    if dp == (c + sum(a))
        necb = floor(log2(k * (2 - 2^(-pin+1))));
    end

    k = k + 1; % Increment test index
end

% Display results
disp(['Detected Number of Extra Carry Bits = ', num2str(necb)]);
disp('===============================================================');
disp('Numerical Feature 6: FMA Size for fp16/bf16 Input');
disp(['Detected FMA Size = ', num2str(NFMA)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Feature 7: Extra Carry Bits in the Accumulator (TF32 Input)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('===============================================================');
disp('Numerical Feature 7: Number of Extra Carry Bits in Accumulator for TF32 Input');

% Input/output format setup
inoptions.format  = 'tf32';
outoptions.format = 'binary32';
inputsize = 16;

% Parameters
pin  = 11;  % Effective mantissa bits for TF32 (10 stored + 1 implicit)
pout = 24;  % fp32 output mantissa bits (including implicit bit)
j    = 0;   % Offset parameter for scaling
k    = 2;   % Starting index for HPEC Algorithm 1
necb = 0;   % Number of extra carry bits (to be detected)

while true
    lgk = ceil(log2(k));

    % Construct test input vectors (carry-propagation detection)
    a = zeros(1, inputsize);
    b = ones(1, inputsize);
    c = 2^j + 2^(-pout+1+j);
    a(1) = 2^j;
    a(k) = 2^(-pout+1+j);

    % Tensor core inner product evaluation
    [dp, dbits, exp, sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
    [dn, dbits, exp, sign] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);

    % Detect termination: FMA block size
    if dp ~= (2^(j+1) + 2^(-pout+2+j)) && abs(dn) ~= (2^(j+1) + 2^(-pout+2+j))
        NFMA = k - 1;
        break;
    end

    % Update for extra carry-bit detection
    a = zeros(1, inputsize);
    b = ones(1, inputsize);
    c = 2 - 2^(-pin+1) + sum(2.^(-pout+1 : -pout+lgk));
    a(1:k-1) = 2 - 2^(-pin+1);
    a(k)     = 2^(-pout+1);

    [dp, dbits, exp, sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);

    % If full propagation occurs, update extra carry bits
    if dp == (c + sum(a))
        necb = floor(log2(k * (2 - 2^(-pin+1))));
    end

    k = k + 1; % Increment test index
end

% Display results
disp(['Detected Number of Extra Carry Bits = ', num2str(necb)]);
disp('===============================================================');
disp('Numerical Feature 8: FMA Size for TF32 Input');
disp(['Detected FMA Size = ', num2str(NFMA)]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Feature 9: Rounding Mode in Compilation of Multiple BFMAs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Objective:
%   Analyze the effective rounding mode when multiple Block Fused Multiply-
%   Add (BFMA) operations are compiled and executed. This test investigates
%   whether the rounding behavior corresponds to the output rounding mode 
%   of a single BFMA or reflects an internal alignment rounding within 
%   individual BFMAs. The analysis uses half-precision inputs and 
%   single-precision outputs.

disp('===============================================================')
disp('Numerical Feature 9: Rounding Mode in Compilation of Multiple BFMAs Results')

%% Floating-point format configuration
inoptions.format  = 'binary16';  % Half-precision inputs
outoptions.format = 'binary32';  % Single-precision outputs

%% BFMA block parameters
NFMA = 8;      % Number of BFMA operations in a block
pout = 24;     % Output precision (mantissa bits for single precision)

%% Reference constant for rounding comparison
j = 0;
c = 2^j + 2^(-pout + 1 + j); % c = 1 + 2^-23, baseline for rounding check

%% Initialize test vectors
a = zeros(1, 2*NFMA);
b = ones(1, 2*NFMA);

% Nontrivial elements to trigger rounding differences
a(NFMA+1) = 2^(-pout + 1 + j);
b(NFMA+1) = 2^(-1) + 2^(-2); % a(NFMA+1)*b(NFMA+1) tests rounding

%% Convert to specified floating-point formats
[a, opts_in] = cpfloat(a, inoptions);
[b, opts_in] = cpfloat(b, inoptions);

%% Simulate BFMA products and rounding
[dp, dbits, exp, sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
[dn, dbits, exp, sign] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);

%% Reference constants for rounding mode detection
rzconst  = c;
rneconst = rzconst + 2^(-pout + j + 1);

%% Determine rounding mode
if dp == rzconst && abs(dn) == rzconst
    disp('Truncation/RZ (Round toward zero)');
elseif dp == rneconst && abs(dn) == rzconst
    disp('RU (Round toward +∞)');
elseif dp == rzconst && abs(dn) == rneconst
    disp('RD (Round toward −∞)');
elseif dp == rneconst && abs(dn) == rneconst
    disp('RNE (Round to nearest even)');
else
    disp('Rounding behavior does not match standard modes');
end

disp('Multiple Block FMA compilation for two BFMAs may reflect internal alignment rounding mode');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Feature 10: Normalisation Pattern Within a BFMA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Objective:
%   Analyze the normalization pattern occurring within a single Block 
%   Fused Multiply-Add (BFMA) operation. This test corresponds to the
%   {1,1} case discussed in Section III C.2 of the referenced HPEC paper.
%   The goal is to determine whether the normalization is immediate or
%   delayed and how it interacts with different rounding modes.

disp('===============================================================')
disp('Numerical Feature 10: Normalisation Pattern Within a BFMA')

%% Floating-point format configuration
inoptions.format  = 'binary16';  % Half-precision inputs
outoptions.format = 'binary32';  % Single-precision outputs

%% Parameters for test
t = 3;                     % Shift parameter for normalization pattern
pout = 24;                  % Output precision (mantissa bits for single)
c = 1 - 2^(-pout + t);      % Reference constant for comparison

%% Initialize test vectors
a = [2^(-pout + t) + 2^-pout, 2^(-pout + t) + 2^-pout, 0, 0, 0, 0, 0, 0];
b = ones(1, 8);             % Vector of ones

%% Convert vectors to specified floating-point formats
[a, opts_in] = cpfloat(a, inoptions);
[b, opts_in] = cpfloat(b, inoptions);

%% Simulate BFMA product with internal normalization
[d, dbits, exp, sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);

%% Determine normalization behavior
if d == (1 + 2^(-pout + t) + 2^(-pout + 1))
    disp('Delayed/Late normalization');
elseif d == (1 + 2^(-pout + t))
    disp('Immediate normalization with RD/RN/TRC/RZ');
elseif d == (1 + 2^(-pout + t) + 2^(-pout + 2))
    disp('Immediate normalization with RU');
else
    disp('Unexpected result: test did not behave as expected');
end

disp('===END===END===END===END===END===END===END===END===END===END===END')




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dec = bin2dec_frac(binStr)
%BIN2DEC_FRAC  Convert a binary string with optional fractional part to decimal.
%   dec = BIN2DEC_FRAC(binStr) interprets binStr, e.g. '1101.101', and returns
%   its decimal equivalent (here 13.625).
%
%   Supports:
%     • Pure-integer strings, e.g. '1011'
%     • Pure-fractional strings, e.g. '.101' (interpreted as 0.625)
%     • Mixed, e.g. '11.01'
%
%   Example:
%     >> bin2dec_frac('1.001')
%     ans =
%        1.125

    % Validate input
    if ~ischar(binStr) && ~isstring(binStr)
        error('Input must be a character vector or string.');
    end
    binStr = char(binStr);
    
    % Split at the decimal point (if any)
    parts = strsplit(binStr, '.');
    switch numel(parts)
        case 1
            intPart = parts{1};
            fracPart = '';
        case 2
            intPart = parts{1};
            fracPart = parts{2};
        otherwise
            error('Invalid binary format: more than one decimal point.');
    end
    
    % Convert integer part (or zero if empty)
    if isempty(intPart)
        intVal = 0;
    else
        % Check for invalid chars
        if any(~ismember(intPart, ['0','1']))
            error('Integer part contains non-binary characters.');
        end
        % Use built‑in for integer portion
        intVal = bin2dec(intPart);
    end
    
    % Convert fractional part (or zero if empty)
    if isempty(fracPart)
        fracVal = 0;
    else
        if any(~ismember(fracPart, ['0','1']))
            error('Fractional part contains non-binary characters.');
        end
        bits = double(fracPart) - '0';
        exponents = -(1:length(bits));
        fracVal = sum(bits .* (2.^exponents));
    end
    
    % Sum
    dec = intVal + fracVal;
end
