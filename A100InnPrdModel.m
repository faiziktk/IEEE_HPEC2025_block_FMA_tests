function [DecOut,BinFixOut,OutExp,OutSignBit ] = A100InnPrdModel(a,b,c,inopts,outopts)
% Developed by FA Khattak, 28/7/25
% takes different rounding mode for alignment, normalisation
% set the rounding mode here in alignment stage

% user variable for multiple stuff
DecOut = 0; % output

% Vary these below parameter to simulate a different model
norm_round='RNE'; % TRC, RU, RD, RNE rounding mode work
align_round='RD'; % TRC, RU, RD, RNE rounding modes
neab=1; % can vary the number of extra carry bits
NFMA=8;

% parameter variable declaration and initialization
params.fma=NFMA;
params.eab=neab;
params.align_round_mode=align_round;
params.norm_round_mode=norm_round;


switch inopts.format
    case {"tf32",'t'}
    NFMA=4; 
    params.fma=NFMA;
        otherwise
end
switch outopts.format
case {"binary16","bfloat16"}
    outopts.round=1;
    params.norm_round_mode='RNE';
otherwise

end

K=numel(a);
if rem(K,NFMA)~=0
rem_elem=mod(K,NFMA);
if rem_elem~=0
    zeropadded=NFMA-rem_elem;
    a=[a,zeros(1,zeropadded)];
    b=[b,zeros(1,zeropadded)];
    
    warning('Next time, enter vector sizes of multiple of 4');
end
end

nFMAs=K/NFMA;
%% nFMAs block FMA operations
for n=1:nFMAs
    [c,BinFixOut,OutExp,OutSignBit] = Generic_BFMA( a( (n-1)*NFMA+1:n*NFMA ), b( (n-1)*NFMA+1:n*NFMA ), c, inopts, outopts,params);
    c=(OutSignBit==1)*(-c)+(OutSignBit==0)*c;
end

DecOut=c;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---------------------  Generic BFMA -----------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [DecOut,BinFixOut,FinalExp,OutSignBit] = Generic_BFMA(a, b, c, inopts, outopts, params)
% a and b are vectors, both either row or column
% assuming round-towards-zero and truncation are alike, not known of any
% example until now
% FMA property of full not including c
  DecOut = 0; % output
  norm_round=params.norm_round_mode;
  align_round=params.align_round_mode;
  neab=params.eab;
  %%------ basic parameters
  % example case
  [~,opts_out] = cpfloat(1, outopts); % products truncated or RZ to out precision
  NoOutExpBits=ceil(log2(opts_out.params(3)))+1; % number of exp bits
  NoOutManBits=opts_out.params(1)-1;         % number of mantissa bits
  OutCharLength=2+NoOutManBits+params.eab;
  
  
  switch outopts.format
      case {'binary16','bfloat16'}
      outopts.format='binary32'; % for products
      otherwise
  end
  %% -----  Input and Product Format conversion
  [a,opts_in] = cpfloat(a, inopts); % products truncated or RZ to out precision
  [b,opts_in] = cpfloat(b, inopts); % products truncated or RZ to out precision
  [r,opts_out] = cpfloat(a.*b, outopts); % products truncated or RZ to out precision
  
  
  %% appending c to the product and then sorting of largest exponent
  r(length(r)+1) = cpfloat(c, outopts); 
  [~,SortIndex]=sort(abs(r),'descend');
  r=r(SortIndex);
 
  %% custom floating point encoder outputting exponent, mantissa, sign and implicit bits
 if NoOutManBits<23
      % assume SP
  [BitCharArray,SignBits,ExpArray,mb_array,imb_array]=CustomFloatEncodeFunc(r,8,23);
 else
      [BitCharArray,SignBits,ExpArray,mb_array,imb_array]=CustomFloatEncodeFunc(r,NoOutExpBits,NoOutManBits);
  end
 %% Significand Alignment and constructing string array
 BitCharArray = significand_alignment_generic(BitCharArray, ExpArray);
 CharLength=numel(BitCharArray(1,:));
 if CharLength<(OutCharLength)
     bitadd=OutCharLength-CharLength;
     zerobits=char( zeros( numel(BitCharArray(:,1)),bitadd) + '0');
     BitCharArray(:,CharLength+1:bitadd+CharLength)=zerobits;
 end
 



if NoOutManBits<23
    [BitStrArray,BitCharArray]= ApplyRoundingPostAlignment(SignBits,BitCharArray,params,23,ExpArray);
    [OutSignBit, resultStr] = fixedPointStringAddSub(SignBits, BitStrArray, 'add'); % addition

else
    [BitStrArray,BitCharArray]= ApplyRoundingPostAlignment(SignBits,BitCharArray,params,NoOutManBits,ExpArray);    
    [OutSignBit, resultStr] = fixedPointStringAddSub(SignBits, BitStrArray, 'add'); % addition
end 

 %% Normalisation procedure keeping extra carry bits and extra alignment bits
 [BinFixOut,FinalExp]=NormalisationPostAddition(resultStr,max(ExpArray));
 
 %% extra bits allowed and then truncation and rounding mode in alignment
%  [OutBit,FinalExp]= round_post_final_normalisation(OutBit,FinalExp,NoOutManBits,norm_round_mode);
 [BinFixOut,OutExp]= ApplyRoundingPostNormalisation(OutSignBit,BinFixOut,params,NoOutManBits,FinalExp);

 DecOut = bin2dec_frac(BinFixOut)*2^FinalExp;
 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Block Alignment Rounding/Truncation As Per Intel Arith Paper 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Calling adding/substraction 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [OutSignBit,resultStr]=RoundPostAlign(BitCharArray,params,NoOutManBits,SignBits)
% some parameters extraction and casting
AlignRoundMode=params.align_round_mode;
neab=params.eab;
NFMA=params.fma;
constparam=2+neab+NoOutManBits;
BitStrArray=string(BitCharArray(:,1:constparam));

% AlignRoundMode checking
if strcmp(AlignRoundMode,'TRC')
                % truncate bits in each significand beyond the extra alignment bit
                BitCharArray(:,1+1+NoOutManBits+neab+1:end)=[];
                BitStrArray=string(BitCharArray);
                % add all signficands at once
                
else
   BitsBeyondAlignBit=numel(BitCharArray(1,2+NoOutManBits+neab+1:end));
         if BitsBeyondAlignBit==0 % similar to truncation
                BitStrArray=string(BitCharArray);
                % add all signficands at once
                [OutSignBit, resultStr] = fixedPointStringAddSub(SignBits, BitStrArray, 'add'); % addition
                return
         elseif BitsBeyondAlignBit==3
               GRT=BitCharArray(2:end,constparam+1:end);
               GRTBitsDec=bin2dec(GRT);
         elseif BitsBeyondAlignBit==2
                GRT=BitCharArray(2:end,constparam+1:end);
                GRT=strcat(GRT,char(zeros(NFMA,1)+'0'));
                GRTBitsDec=bin2dec(GRT);
         elseif BitsBeyondAlignBit==1
               GRT=BitCharArray(2:end,constparam+1:end);
               GRT=strcat(GRT,char(zeros(NFMA,2)+'0'));
               GRTBitsDec=bin2dec(GRT);
         else 
               GRT=BitCharArray(2:end,constparam+1:constparam+2); % GR bits
               GRT=strcat(GRT,char(zeros(NFMA,1)+'0'));
               GRTBitsDec=bin2dec(GRT);
               stickyBit=BitCharArray(2:end,constparam+4:end);
               stickyBitDec=bin2dec(stickyBit);
               stickyBitDec(stickyBitDec>1)=1;
               GRTBitsDec=GRTBitsDec+stickyBitDec;
         end
           % GRT bits detection and extraction complete above
           
           InpStrArray=string(BitStrArray{1});
           SignA=SignBits(1);
           
           for n=1:NFMA
            InpStrArray{2}=BitStrArray{n+1};
            InpSignBits=[SignA,SignBits(n+1)];
            [SignA, resultStr] = fixedPointStringAddSub(InpSignBits, InpStrArray, 'add'); % addition
            resultStr = ScalarFMARoundingPreNorm(resultStr, GRTBitsDec(n), AlignRoundMode, SignA);          
            InpStrArray=string(resultStr);
           end 
           OutSignBit=SignA;

 end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   bin2dec_fraction function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Normalisation function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [d_in_bits,final_exp_actual]=NormalisationPostAddition(resultStr,largest_exp) 

dotPos = strfind(resultStr, '.');
if isempty(dotPos)
    error('Input must contain a binary point. Error in function normalisation_proc');
end

 charCount=numel(resultStr);
 exp_shift_result=0;
% checking if integer bits has increased due to carry in MSBs
 split_array=split(resultStr,'.');
 bit_before_dec_point=numel(split_array{1});
 % carry occured
 integer_part=bin2dec(split_array{1});
if integer_part==1
    % no change
    d_in_bits=resultStr;

elseif integer_part>1
    % shift to the right and increase the exponent
  exp_shift_result=bit_before_dec_point-1;
  d_in_bits=strcat('1','.',split_array{1}(2:end),split_array{2});
else
% small and therefore left shift and decrease the exponent
    firstOne = find(split_array{2} == '1', 1, 'first');
    exp_shift_result=-firstOne;
    
    if isempty(firstOne)
    % all zeros no firstOne
        d_in_bits=resultStr; % no change
        final_exp_actual=largest_exp;
    else
         d_in_bits=strcat('1.',split_array{2}(firstOne+1:end));
         extracharappend=charCount-numel(d_in_bits);
         d_in_bits=strcat(d_in_bits,char(zeros(1,extracharappend)+'0'));
    end

    
end

 
 final_exp_actual=largest_exp+exp_shift_result;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%----------- Rounding Post Alignment Post Single Addition ---------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rounded = ScalarFMARoundingPreNorm(bitstr, grt, roundmode, signBit)
% ScalarFMARoundingPreNorm  Round a fixed-precision binary string with GRS bits
%
% INPUTS:
%   bitstr    - binary string with decimal point (e.g. '1101.1011')
%   grt       - decimal representation of 3-bit GRS (e.g. 5 for '101')
%   roundmode - 'RNE' = round to nearest, ties-to-even
%               'RU'  = round toward +inf
%               'RD'  = round toward -inf
%   signBit   - 0 for positive, 1 for negative
%
% OUTPUT:
%   rounded   - rounded binary string (same length as bitstr)

    % Split integer and fractional parts
    parts = split(bitstr, '.');
    intPart = parts{1};
    if numel(parts) > 1
        fracPart = parts{2};
    else
        fracPart = '';
    end

    % Convert GRS decimal to binary (3-bit string: G,R,S)
    grs_bits = dec2bin(grt, 3);
    G = str2double(grs_bits(1));
    R = str2double(grs_bits(2));
    S = str2double(grs_bits(3));

    % Default decision
    roundUp = false;

    switch upper(roundmode)
        case 'RNE' % Round to nearest, ties-to-even
            if (G == 1 && (R == 1 || S == 1)) % strictly greater than half
                roundUp = true;
            elseif (G == 1 && R == 0 && S == 0) % exactly half
                % Tie: check LSB (last kept bit)
                if isempty(fracPart)
                    LSB = str2double(intPart(end));
                else
                    LSB = str2double(fracPart(end));
                end
                if mod(LSB, 2) == 1
                    roundUp = true; % make even
                end
            end

        case 'RPI' % Round toward +inf
            if signBit == 0 % positive: round up if any discarded bits
                if (G == 1 || R == 1 || S == 1)
                    roundUp = true;
                end
            else % negative: truncate
                roundUp = false;
            end

        case 'RNI' % Round toward -inf
            if signBit == 0 % positive: truncate
                roundUp = false;
            else % negative: more negative => increment if any discarded bits
                if (G == 1 || R == 1 || S == 1)
                    roundUp = true;
                end
            end

        otherwise
            error('Unknown rounding mode. Use RNE, RU, or RD.');
    end

    % Apply rounding
    if roundUp
        if isempty(fracPart)
            % Only integer part
            intPart = addBinary(intPart, '1');
        else
            % Add 1 to fractional part (LSB of kept precision)
            [fracPart, carry] = addBinaryFixed(fracPart, '1');
            if carry
                intPart = addBinary(intPart, '1');
            end
        end
    end

    % Return same-precision result
    if isempty(fracPart)
        rounded = intPart;
    else
        rounded = [intPart '.' fracPart];
    end
end

function sumStr = addBinary(a, b)
% Add binary strings with variable length
    a_dec = bin2dec(a);
    b_dec = bin2dec(b);
    sumStr = dec2bin(a_dec + b_dec);
end

function [outStr, carry] = addBinaryFixed(a, b)
% Add binary string a + b with fixed length = length(a)
    a_dec = bin2dec(a);
    b_dec = bin2dec(b);
    sumVal = a_dec + b_dec;
    maxVal = 2^length(a);
    if sumVal >= maxVal
        carry = 1;
        sumVal = sumVal - maxVal;
    else
        carry = 0;
    end
    outStr = dec2bin(sumVal, length(a)); % keep same length
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%----------- Normalisation after rounding with carry in MSBs -------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,expt]=ApplyNormalisationPostRounding(temp_add_return,out_nmb,extra_align_bit,expt)

%new_grt_bits=temp_add_return(out_nmb+2+extra_align_bit+1:end);
%new_grt_dec=bin2dec(new_grt_bits);
str_sp=split(temp_add_return,'.');
if numel(str_sp{1})>1
% carry has occurred due to rounding
expt=expt+1;
x=strcat(temp_add_return(1),'.',temp_add_return(2),temp_add_return(4:end-1));

else
% no carry, no normalisation required
x=temp_add_return;
% no exponent adjusting
end



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%----------- New Alignment Function -------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function c_shifted = significand_alignment_generic(c, e)
    % c: N x M char array, each row is a binary string like '1.001'
    % e: N x 1 vector of exponents
    % Output: c_shifted: N x L char array, each row is a padded binary string

    N = size(c, 1);
    max_exp = max(e);
    temp = cell(N,1);
    max_len = numel(c(1,:));
    temp{1} = c(1,:);
    % First pass: build shifted strings and find max length
    for i = 2:N
        str = c(i,:);
        parts = strsplit(str, '.');
        int_part = parts{1};
        if numel(parts) > 1
            frac_part = parts{2};
        else
            frac_part = '';
        end

        full_bits = [int_part frac_part];
         
        shift_amt = max_exp - e(i);
        ddec=bin2dec_frac(c(i,:));
        if shift_amt==0 | ddec==0
            % no change
            temp{i}=c(i,:);

        else
            shifted_str = ['0.' repmat('0', 1, shift_amt-1) full_bits];
            temp{i} = shifted_str;
            max_len = max(max_len, length(shifted_str));
        end
            
    end

    % Second pass: pad strings to equal length
    c_shifted = char(zeros(N, max_len) + ' ');
    for i = 1:N
        str = temp{i};
        str_padded = [str repmat('0', 1, max_len - length(str))];
        c_shifted(i, :) = str_padded;
    end
end

%%#########################################################################
%% Function Rounding Mode Post Normalisation
%% ########################################################################
function [BitCharArray,OutExp]= ApplyRoundingPostNormalisation(SignBits,BitCharArray,params,out_nmb,Expts)
RoundMode=params.norm_round_mode;
CharLength=numel(BitCharArray(1,:));
OutCharLength=2+out_nmb;
OutExp=Expts;
% Extract GRT bits in advance
if CharLength>OutCharLength
grtBits=BitCharArray(:,OutCharLength+1:end);
else
% nothing
return
end
BitCharArray(:,OutCharLength+1:end)=[]; % discard bit beyond output precision

%% --------------- 1: Round towards Zero or Truncation ----------------------
if strcmp(RoundMode,'TRC')|strcmp(RoundMode,'RTZ') 
    % BitStringArray=string(BitCharArray);
     return
end    

%% --------------- 2: Round towards Postive Infinity ----------------------
ZeroCharsArray=char(zeros(1,OutCharLength-3)+'0');
if strcmp(RoundMode,'RU') % Round Positive Infinity
    
    ZeroCharsArray=char(zeros(1,OutCharLength-3)+'0');
    
    
            if bin2dec(grtBits)>0
                % sign check
                if SignBits
                 % nothing happens bcz its similar to truncation
                else % positive case

                    TempInpStr{1}=BitCharArray;
                    TempInpStr{2}=strcat('0.',ZeroCharsArray,'1');
                    [~,TempOutCharArray] = fixedPointStringAddSub([0,0],TempInpStr, 'add'); % addition
                    % may need normalisation just because of rounding 
                    [BitCharArray,OutExp]=ApplyNormalisationPostRounding(TempOutCharArray,out_nmb,params.eab,Expts);
                    
                end
                    
        
            end
  %  BitStringArray=string(BitCharArray);

return
end
%% -------------------------------------------------------------------------

%% ------------------------------------------------------------------------
%% --------------- 3: Round towards Negative Infinity ----------------------
%% ------------------------------------------------------------------------

if strcmp(RoundMode,'RD') % Round Positive Infinity
    
    
    
    
        if bin2dec(grtBits)>0
                % sign check
                if ~SignBits
                 % nothing happens bcz its similar to truncation but for
                 % positive number
                else % negative case

                    TempInpStr{1}=BitCharArray;
                    TempInpStr{2}=strcat('0.',ZeroCharsArray,'1');
                    [~,TempOutCharArray] = fixedPointStringAddSub([0,0],TempInpStr, 'add'); % addition
                    % may need normalisation just because of rounding 
                    [BitCharArray,OutExp]=ApplyNormalisationPostRounding(TempOutCharArray,out_nmb,params.eab,Expts);
                    
                end
                    
        end
    
    %BitStringArray=string(BitCharArray);

return
end
%% -------------------------------------------------------------------------

%% --------------- 3: Round towards Nearest Ties to Even-------------------


if RoundMode=='RNE' % round to zero even
    
    if numel(grtBits)>3
        stickyBit=grtBits(:,3:end);
        grtBits(:,3:end)=[];% 
        grtBits=strcat(grtBits,char(zeros(1,1)+'0'));
        grtBitsDec=bin2dec(grtBits);% 

        stickyBitDec=bin2dec(stickyBit);
        stickyBitDec(stickyBitDec>1)=1;
        grtBitsDec=grtBitsDec+stickyBitDec;
    else
        if ~isempty(grtBits)
            nbits=numel(grtBits);
            if nbits<3
                grtBits=strcat(grtBits,char(zeros(1,3-nbits)+'0'));
            end
        grtBitsDec=bin2dec(grtBits);
        end
    end

    if ~isempty(grtBits)
         
         
             if (grtBitsDec==4 & BitCharArray(end)=='1') | grtBitsDec>4
                    TempInpStr{1}=BitCharArray;
                    TempInpStr{2}=strcat('0.',ZeroCharsArray,'1');
                    [~,TempOutCharArray] = fixedPointStringAddSub([0,0],TempInpStr, 'add'); % addition
                    % may need normalisation just because of rounding 
                    [BitCharArray,OutExp]=ApplyNormalisationPostRounding(TempOutCharArray,out_nmb,params.eab,Expts);

             end
         
         
         

    end
%BitStringArray=string(BitCharArray);
return

end

%BitStringArray=string(BitCharArray);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [BitCharArray,SignBits,ExpArray,mb_array,imb_array]=CustomFloatEncodeFunc(r,NoOutExpBits,NoOutManBits)

  for i=1:numel(r)
    [SignBits(i), eb_array(i,:), mb_array(i,:), imb_array(i,1)] = customFloatEncode(r(i), NoOutExpBits, NoOutManBits);
    % exponents extractions
    ExpArray(i)= bin2dec(char(eb_array(i,:)+ '0'))-(2^(NoOutExpBits-1)-1);
    % character array extraction
    BitCharArray(i,:)=strcat(num2str(imb_array(i,1)),'.',char(mb_array(i,:)+'0'));
 
  end
end


%%#########################################################################
%% Function Rounding Mode Post Alignment
%% ########################################################################
function [BitStringArray,BitCharArray]= ApplyRoundingPostAlignment(SignBits,BitCharArray,params,out_nmb,Expts)
RoundMode=params.align_round_mode;
CharLength=numel(BitCharArray(1,:));
OutCharLength=2+out_nmb+params.eab;
K=numel(BitCharArray(:,1));
% Extract GRT bits in advance
if CharLength>OutCharLength
grtBits=BitCharArray(:,OutCharLength+1:end);
else
% nothing
BitStringArray=string(BitCharArray);
return
end
BitCharArray(:,OutCharLength+1:end)=[]; % discard bit beyond output precision

%% --------------- 1: Round towards Zero or Truncation ----------------------
if strcmp(RoundMode,'TRC')|strcmp(RoundMode,'RZ') 
     BitStringArray=string(BitCharArray);
     return
end    

%% --------------- 2: Round towards Postive Infinity ----------------------
ZeroCharsArray=char(zeros(1,OutCharLength-3)+'0');
if strcmp(RoundMode,'RU') % Round Positive Infinity
    
    ZeroCharsArray=char(zeros(1,OutCharLength-3)+'0');
    
    for k=1:K
            if bin2dec(grtBits(k,:))>0
                % sign check
                if SignBits(k)
                 % nothing happens bcz its similar to truncation
                else % positive case

                    TempInpStr{1}=BitCharArray(k,:);
                    TempInpStr{2}=strcat('0.',ZeroCharsArray,'1');
                    [~,TempOutCharArray] = fixedPointStringAddSub([0,0],TempInpStr, 'add'); % addition
                    % may need normalisation just because of rounding 
                    [BitCharArray(k,:),~]=ApplyNormalisationPostRounding(TempOutCharArray,out_nmb,params.eab,Expts(k));
                    
                end
                    
        end
    end
    BitStringArray=string(BitCharArray);

return
end
%% -------------------------------------------------------------------------

%% ------------------------------------------------------------------------
%% --------------- 3: Round towards Negative Infinity ----------------------
%% ------------------------------------------------------------------------

if strcmp(RoundMode,'RD') % Round Positive Infinity
    
    
    
    for k=1:K
        if bin2dec(grtBits(k,:))>0
                % sign check
                if ~SignBits(k)
                 % nothing happens bcz its similar to truncation but for
                 % positive number
                else % negative case

                    TempInpStr{1}=BitCharArray(k,:);
                    TempInpStr{2}=strcat('0.',ZeroCharsArray,'1');
                    [~,TempOutCharArray] = fixedPointStringAddSub([0,0],TempInpStr, 'add'); % addition
                    % may need normalisation just because of rounding 
                    [BitCharArray(k,:),~]=ApplyNormalisationPostRounding(TempOutCharArray,out_nmb,params.eab,Expts(k));
                    
                end
                    
        end
    end
    BitStringArray=string(BitCharArray);

return
end
%% -------------------------------------------------------------------------

%% --------------- 3: Round towards Nearest Ties to Even-------------------


if RoundMode=='RNE' % round to zero even
    
    if numel(grtBits)>3
        stickyBit=grtBits(:,3:end);
        grtBits(:,3:end)=[];% 
        grtBits=strcat(grtBits,char(zeros(K,1)+'0'));
        grtBitsDec=bin2dec(grtBits);% 
        stickyBitDec=bin2dec(stickyBit);
        stickyBitDec(stickyBitDec>1)=1;
        grtBitsDec=grtBitsDec+stickyBitDec;
    else
        if ~isempty(grtBits)
            nbits=numel(grtBits(1,:));
            if nbits<3
              zeroappend=char(zeros(numel(grtBits(:,1)),3-nbits)+'0');
              grtBits=strcat(grtBits,zeroappend);
            end
        grtBitsDec=bin2dec(grtBits);
        end
        
    end

    if ~isempty(grtBits)
         for k=1:K
         
             if (grtBitsDec(k)==4 & BitCharArray(k,end)=='1') | grtBitsDec(k)>4
                    TempInpStr{1}=BitCharArray(k,:);
                    TempInpStr{2}=strcat('0.',ZeroCharsArray,'1');
                    [~,TempOutCharArray] = fixedPointStringAddSub([0,0],TempInpStr, 'add'); % addition
                    % may need normalisation just because of rounding 
                    [BitCharArray(k,:),]=ApplyNormalisationPostRounding(TempOutCharArray,out_nmb,params.eab,Expts(k));

             end
         
         
         end

    end
BitStringArray=string(BitCharArray);
return

end

BitStringArray=string(BitCharArray);


end
