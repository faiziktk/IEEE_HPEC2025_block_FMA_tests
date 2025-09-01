clear all;
close all
clc
% Example by FA Khattak 18/7/2025
% Testing A100/Ada 1000 Tensor Dot Product Model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up input and output formats.
% Note the default rounding mode in CPFloat is RN-TE.
inoptions.format = 'binary16'; % 
outoptions.format = 'binary32';
%% 
% This test checks the number of extra alignment bits
% Only two bits are tested, for more bits, refer to HPEC'25 paper 
% Faizan AK, M Mikaitis
test_bool=1;
nmb=23; % single precision
[a, opts_in] = cpfloat([2^(-24),2^(-24),0,0,0,0,0,0], inoptions);
[b, opts_in] = cpfloat([1,1,1,1,1,1,1,1], inoptions);
[c, opts_out] = cpfloat(1, outoptions);
pout = opts_out.params(1)-1;
[d,dbits,exp,sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
bool_eab_test_1=strcmp(strcat('1.',char(zeros(1,nmb-1)+'0'),'1'),dbits);

[a, opts_in] = cpfloat([2^(-24),2^(-24),2^(-24),0,0,0,0,0], inoptions);
[b, opts_in] = cpfloat([2^-1,2^-1,1,1,1,1,1,1], inoptions);
pin = opts_in.params(1)-1; % as per our definition
[d,dbits,exp,sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
bool_eab_test_2= (d==(a*b'+c))
clc
disp('HPEC-25 Tensor Core Matrix Multipliers Numerical Feature Testing')
disp('Authors: Faizan A. Khattak, Mantas Mikaitis')
disp('===============================================================')
disp('Numerical Feature 1: Extra Alignment Bit Test')
if bool_eab_test_1==1
disp('One extra bit in accumulator exists');
else
    disp('No extra bits exist in alignment');

end
if bool_eab_test_2==1
disp('Two bits exist in alignment, necb>=2');
else
disp('Two bits does not exist in alignment');    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rounding/Truncation at Alignment of Significands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% assuming the model has one extra alignment bit
% input bf16/bf16 output binary32 
disp('===============================================================')
disp('Numerical Feature 2: Rounding/Truncation of Aligned Significands')

[a, opts_in] = cpfloat([2^(-23),2^-23,0,0,0,0,0,0], inoptions);
[b, opts_in] = cpfloat([sum(2.^(-1:-1:-3)),sum(2.^(-1:-1:-3)),1,1,1,1,1,1], inoptions);
[c, opts_out] = cpfloat(1, outoptions);
pout = opts_out.params(1)-1;
[dp,dbits,exp,signp] = A100InnPrdModel(a, b, c, inoptions, outoptions);
[dn,dbits,exp,signn] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);

if dp==(1+2^-22) & abs(dn)==(1+2^-22)
    disp('RNE');
elseif dp==(1+2^-23) & abs(dn)==(1+2^-23)
    disp('Truncation');
elseif dp==(1+2^-22) & abs(dn)==(1+2^-23)
     disp('RU');
elseif dp==(1+2^-23) & abs(dn)==(1+2^-22)
     disp('RD');
else

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Final Output Rounding Mode
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% no possibility of earlier rounding considering block FMA property
% disp('-----------------------------------------');
disp('===============================================================')
disp('Numerical Feature 3: Final Output Rounding Mode in fp32 Output Mode')
[a, opts_in] = cpfloat([1,1,1,2^-23,2^-22,0,0,0], inoptions);
[b, opts_in] = cpfloat([1,1,1,1,1,1,1,1], inoptions);
[c, opts_out] = cpfloat(1, outoptions);
[dp,dbits,exp,sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
[dn,dbits,exp,sign] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);

if dp==(4+2^-21) & abs(dn)==(4+2^-21)
    disp('RNE');
elseif dp==(4) & abs(dn)==(4)
    disp('Truncation');
elseif dp==(4+2^-21) & abs(dn)==(4)
     disp('RU');
elseif dp==(4) & abs(dn)==(4+2^-21)
     disp('RD');
else

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Final Output Rounding Mode
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% no possibility of earlier rounding considering block FMA property
outoptions.format = 'binary16';
disp('===============================================================')
disp('Numerical Feature 4: Final Output Rounding Mode in fp16/bf16 Output Mode')
[a, opts_in] = cpfloat([0,0,0,2^-11,2^-12,0,0,0], inoptions);
[b, opts_in] = cpfloat([1,1,1,1,1,1,1,1], inoptions);
[c, opts_out] = cpfloat(1, outoptions);
[dp,dbits,exp,sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
[dn,dbits,exp,sign] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);

if dp==(1+2^-10) & abs(dn)==(1+2^-10)
    disp('Round-to-Nearest (Ties to Even)');
elseif dp==(1) & abs(dn)==(1)
    disp('Truncation');
elseif dp==(1+2^-10) & abs(dn)==(1)
     disp('RU');
elseif dp==(1) & abs(dn)==(1+2^-10)
     disp('RD');
else

end


%%--------------------------------------------------------------------
%% --- Number Extra Carry Bits  independent of FMA Size---------------
%%--------------------------------------------------------------------
disp('===============================================================')
disp('Numerical Feature 5: Number of Extra Carry Bits in Accumulator for fp16/bf16 Input')
inoptions.format = 'binary16';
outoptions.format = 'binary32';
inputsize=16;
pin=11; % fp16 input
pin=pin-1;
pout=24;% fp32 output, assuming 1 implicit bit in there
pout=pout-1;
k=2; % HPEC paper Algorithm 1
flag=1;
counter=1;
while 1

a=zeros(1,inputsize);
b=ones(1,inputsize);
c=1+2^(-pout);
a( 1 )=1;
a(k)=2^(-pout);
[dp,dbits,exp,sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
[dn,dbits,exp,sign] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);

%d = bin2dec_frac(dbits)*2^(exp);
if dp~=(2+2^(-pout+1)) & abs(dn)~=(2+2^(-pout+1))
     NFMA=k-1;
     necb=log2(ceil(2*NFMA/(2-2^(-pin+1)))-1);
     break;
end
k=k+1;
counter=counter+1;
end
disp(strcat('Number of Extra Carry Bits=',num2str(necb)));
disp('===============================================================')
disp('Numerical Feature 6: FMA Size for fp16/bf16 Input')
disp(strcat('The FMA size is=',num2str(NFMA)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
neab=[];
disp('===============================================================')
disp('Numerical Feature 7: Number of Extra Carry Bits in Accumulator for tf32 Input')
inoptions.format = 'tf32';
outoptions.format = 'binary32';
inputsize=16;
pin=11; % fp16 input
pin=pin-1;
pout=24;% fp32 output, assuming 1 implicit bit in there
pout=pout-1;
k=2; % HPEC paper Algorithm 1
flag=1;
counter=1;
while flag
     a=zeros(1,inputsize);
     b=ones(1,inputsize);
     c=1+2^(-pout);
     a( 1 )=1;
     a(k)=2^(-pout);
    [dp,dbits,exp,sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
    [dn,dbits,exp,sign] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);

%d = bin2dec_frac(dbits)*2^(exp);
if dp~=(2+2^(-pout+1)) & abs(dn)~=(2+2^(-pout+1))
     NFMA=k-1;
     necb=log2(ceil(2*NFMA/(2-2^(-pin+1)))-1);
     break;
end
k=k+1;
end

disp(strcat('Number of Extra Carry Bits= ',num2str(necb)));
disp('===============================================================')
disp('Numerical Feature 8: FMA Size for tf32 Input')

disp(strcat('The FMA size is=',num2str(NFMA)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rounding Mode for Multiple Block FMA results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A model with 2 or mode extra alignment bits, this numerical feature will be
% same as output mode of a single BFMA, otherwise, the rounding mode
% reflected may be of the alignment roundign mode within a single BFMA
disp('===============================================================')
disp('Numerical Feature 9: Rounding Mode in Compilation of Multiple BFMAs Results')
inoptions.format = 'binary16';
outoptions.format = 'binary32';
NFMA=8; % obtained previously
pout=24;
c=1+2^(-pout+1); % c=1+2^-23
a=zeros(1,2*NFMA);
b=ones(1,2*NFMA);
a(NFMA+1)=2^(-pout+1);
b(NFMA+1)=2^(-1)+2^(-2); % a(NFMA+1)*b(NFMA+1)=r(NFMA+1)=2^(-pout)+2^(-pout-1)
[a, opts_in] = cpfloat(a, inoptions);
[b, opts_in] = cpfloat(b, inoptions);
[dp,dbits,exp,sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
[dn,dbits,exp,sign] = A100InnPrdModel(-a, b, -c, inoptions, outoptions);
if dp==(1+2^(-pout+1)) & abs(dn)==(1+2^(-pout+1))
    disp('Truncation/RZ');
elseif dp==(1+2^(-pout+2)) & abs(dn)==(1+2^(-pout+1))
    disp('RU');
elseif dp==(1+2^(-pout+1)) & abs(dn)==(1+2^(-pout+2))
     disp('RD');
elseif dp==(1+2^(-pout+2)) & abs(dn)==(1+2^(-pout+2))
     disp('RNE');
else

end
disp('Multiple Block FMA compilation for two BFMA may have alignment rounding mode');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Normalisation Pattern
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('===============================================================')
disp('Numerical Feature 10: Normalisation Pattern Within a BFMA')
%{1,1} case of the HPEC paper in Sec.III C.2
inoptions.format = 'binary16';
outoptions.format = 'binary32';
t=3;
pout=24;
c=1-2^(-pout+t);
a=[2^(-pout+t)+2^-pout,2^(-pout+t)+2^-pout,0,0,0,0,0,0];
b=[1,1,1,1,1,1,1,1];
[a, opts_in] = cpfloat(a, inoptions);
[b, opts_in] = cpfloat(b, inoptions);

[d,dbits,exp,sign] = A100InnPrdModel(a, b, c, inoptions, outoptions);
if d==(1+2^(-pout+t)+2^(-pout+1))
        disp('Delayed/Late normalisation')
elseif d==(1+2^(-pout+t))
        disp('Immediate normalisation with RD/RN/TRC/RZ');
elseif d==(1+2^(-pout+t)+2^(-pout+2))
        disp('Immediate normalisation with RU');
else
    disp('Bro! your test is not working, lol')
end

disp('===============================================================')
disp('===============================================================')




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
