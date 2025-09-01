function [sOut, resultStr] = fixedPointStringAddSub(signs, binStrs, op)
% Inputs:
%   signs     - Nx1 vector of sign bits (0 = +, 1 = -)
%   binStrs   - Nx1 cell array of binary strings with fixed frac (e.g., '1.101')
%   op        - 'add' or 'sub'
%
% Outputs:
%   sOut      - Sign bit of result
%   resultStr - Binary string with same number of fractional bits

N = length(binStrs);

% Determine number of fractional bits
parts = split(binStrs{1}, '.');
fracBits = strlength(parts{2});

% Convert all binary strings to decimal numbers
values = zeros(N,1);
for i = 1:N
    parts = split(binStrs{i}, '.');
    intPart = bin2dec(parts{1});
    fracStr = parts{2};
    
    fracPart = 0;
    for j = 1:fracBits
        if fracStr(j) == '1'
            fracPart = fracPart + 2^(-j);
        end
    end
    
    val = intPart + fracPart;
    values(i) = (-1)^signs(i) * val;
end

% Perform operation
if strcmp(op, 'add')
    result = sum(values);
elseif strcmp(op, 'sub')
    result = values(1) - sum(values(2:end));
else
    error('Invalid operation. Use "add" or "sub".');
end

% Determine sign bit and absolute value
sOut = result < 0;
result = abs(result);

% Split into integer and fractional parts
intPart = floor(result);
fracPart = result - intPart;

% Convert integer part to binary
intStr = dec2bin(intPart);

% Convert fractional part to fixed number of binary digits
fracStr = '';
for j = 1:fracBits
    fracPart = fracPart * 2;
    bit = floor(fracPart);
    fracStr = strcat(fracStr, num2str(bit));
    fracPart = fracPart - bit;
end

% Construct final binary string
resultStr = [intStr '.' fracStr];
end
