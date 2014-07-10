function model = vaw_train(X,Y,model)
% VAW_TRAIN Vovk–Azoury–Warmuth forecaster 
%
%    MODEL = VAW_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Vovk–Azoury–Warmuth algorithm.
%
%    Additional parameters:
%    - model.a is the regularization parameter.
%      Default value is 1.
%
%   References:
%     - Vovk., V. (2001).
%       Competitive on-line statistics.
%       International Statistical Review, 69, (pp. 213-248). 
%
%     - Azoury, K. S., & Warmuth, M. (2001).
%       Relative loss bounds for on-line density estimation with the
%       exponential family of distributions.
%       Machine Learning, 43(3), (pp. 211-246).

%    This file is part of the DOGMA library for MATLAB.
%    Copyright (C) 2009-2011, Francesco Orabona
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    Contact the author: francesco [at] orabona.com

n = length(Y);

if isfield(model,'a')==0
    model.a = 1;
end

if isfield(model,'iter')==0
    model.iter = 0;
    model.w = zeros(1,size(X,1));
    model.w2 = zeros(1,size(X,1));
    model.errTot = 0;
    model.numSV = zeros(numel(Y),1);
    model.aer = zeros(numel(Y),1);
    model.pred = zeros(numel(Y),1);

    model.KbInv = model.a*eye(size(X,1));
end

for i = 1:n
    model.iter = model.iter+1;
        
    val_f = model.w*X(:,i);

    KbInv_x = model.KbInv*X(:,i);
    r = X(:,i)'*KbInv_x;
    
    % Include the current sample to predict
    val_f = val_f/(r+1);
    
    Yi = Y(i);
    
    model.errTot = model.errTot+(val_f-Yi)^2;
    model.aer(model.iter) = model.errTot/model.iter;
    model.pred(model.iter) = val_f;

    model.w = model.w+(Yi-val_f)/(1+r)*KbInv_x';
    model.KbInv = model.KbInv-KbInv_x*KbInv_x'/(1+r);

    model.S(end+1) = model.iter;

    model.w2 = model.w2+model.w;
    
    model.numSV(model.iter) = numel(model.S);

    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end
