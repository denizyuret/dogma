function model = narow_train(X,Y,model)
% NAROW_TRAIN Narrow Adaptive Regularization Of Weights algorithm
%
%    MODEL = NAROW_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Narrow Adaptive Regularization Of Weights algorithm.
%
%    Additional parameters:
%    - model.a is the parameter to control to growth of the covariance
%      matrix
%      Default value is 1.
%
%   References:
%     - Orabona, F., & Crammer, K. (2010).
%       New Adaptive Algorithms for Online Classiﬁcation.
%       In Advances in Neural Information Processing Systems 23 (NIPS10).

%    This file is part of the DOGMA library for Matlab.
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

n = length(Y);   % number of training samples

if isfield(model,'iter')==0
    model.iter = 0;
    model.w = zeros(1,size(X,1));
    model.w2 = zeros(1,size(X,1));
    model.errTot = 0;
    model.numSV = zeros(numel(Y),1);
    model.aer = zeros(numel(Y),1);
    model.pred = zeros(numel(Y),1);

    model.theta = zeros(1,size(X,1));
    model.KbInv = eye(size(X,1));
end

if isfield(model,'a')==0
    model.a = 1;
end

for i=1:n
    model.iter=model.iter+1;

    val_f = model.w*X(:,i);

    Yi = Y(i);
    
    KbInv_x = model.KbInv*X(:,i);
    chi = X(:,i)'*KbInv_x;

    if chi>=1/model.a
        r_t = chi/(model.a*chi-1);
        val_f = r_t*val_f/(chi+r_t);
    else
        r_t = inf;
    end

    model.errTot = model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter) = model.errTot/model.iter;
    model.pred(model.iter) = val_f;
                
    if Yi*val_f<1
        model.S(end+1) = model.iter;
        
        model.theta = model.theta+Yi*X(:,i)';
        if r_t<inf
            model.KbInv = model.KbInv-KbInv_x*KbInv_x'/(r_t+chi);
        end
        model.w = model.theta*model.KbInv;
    end
    
    model.numSV(model.iter) = numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end