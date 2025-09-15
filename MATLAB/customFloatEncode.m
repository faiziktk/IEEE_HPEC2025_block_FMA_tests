function [signBit, exponentBits, mantissaBits, hasImplicitOne] = customFloatEncode(x, expBits, mantBits)
%CUSTOMFLOATENCODE Encode a real number into custom floating-point bits
%
% Inputs:
%   x        - floating-point number
%   expBits  - number of exponent bits (e.g., 5)
%   mantBits - number of mantissa bits (e.g., 10)
%
% Outputs:
%   signBit        - 0 or 1
%   exponentBits   - vector of exponent bits (length = expBits)
%   mantissaBits   - vector of mantissa bits (length = mantBits)
%   hasImplicitOne - true if leading 1 is implicit (normal), false otherwise

% Handle sign
if isnan(x)
    signBit = 0; % Arbitrary
elseif x < 0
    signBit = 1;
    x = -x;
else
    signBit = 0;
end

% Constants
bias = 2^(expBits - 1) - 1;
maxExpVal = 2^expBits - 1;

% Handle special cases
if isnan(x)
    exponentBits = ones(1, expBits);
    mantissaBits = [1, zeros(1, mantBits - 1)]; % Quiet NaN
    hasImplicitOne = false;
    return
elseif isinf(x)
    exponentBits = ones(1, expBits);
    mantissaBits = zeros(1, mantBits);
    hasImplicitOne = false;
    return
elseif x == 0
    exponentBits = zeros(1, expBits);
    mantissaBits = zeros(1, mantBits);
    hasImplicitOne = false;
    return
end

% Compute raw exponent
expRaw = floor(log2(x));
frac = x / 2^expRaw;

% Normalize mantissa
if frac < 1
    expRaw = expRaw - 1;
    frac = frac * 2;
end

% Bias the exponent
expBiased = expRaw + bias;

if expBiased <= 0
    % Subnormal number
    exponentBits = zeros(1, expBits);
    frac = x / 2^(1 - bias); % shift to subnormal range
    hasImplicitOne = false;
else
    if expBiased >= maxExpVal
        % Overflow → Inf
        exponentBits = ones(1, expBits);
        mantissaBits = zeros(1, mantBits);
        hasImplicitOne = false;
        return
    end
    exponentBits = bitget(uint32(expBiased), expBits:-1:1);
    frac = frac - 1; % remove implicit 1
    hasImplicitOne = true;
end

% Generate mantissa bits
mantissaBits = zeros(1, mantBits);
for i = 1:mantBits
    frac = frac * 2;
    if frac >= 1
        mantissaBits(i) = 1;
        frac = frac - 1;
    end
end
end
