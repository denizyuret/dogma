function model = arow_train(X,Y,model)
% AROW_TRAIN Adaptive Regularization Of Weights algorithm
%
%    MODEL = AROW_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Adaptive Regularization Of Weights algorithm.
%
%    Additional parameters:
%    - model.r is the learning rate parameter.
%      Default value is 1.
%
%   References:
%     - Crammer, K., Kulesza, A., & Dredze, M. (2009)
%       Adaptive Regularization Of Weight Vectors.
%       Advances in Neural Information Processing Systems

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

    model.KbInv = eye(size(X,1));
end

if isfield(model,'r')==0
    model.r = 1;
end

for i=1:n
    model.iter=model.iter+1;

    val_f = model.w*X(:,i);

    Yi = Y(i);
    
    model.errTot = model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter) = model.errTot/model.iter;
    model.pred(model.iter) = val_f;
                
    if Yi*val_f<1
        model.S(end+1) = model.iter;
        KbInv_x=model.KbInv*X(:,i);
        chi=X(:,i)'*KbInv_x;
        
        model.w=model.w+(Yi-val_f)/(1+chi)*KbInv_x';
        model.KbInv=model.KbInv-KbInv_x*KbInv_x'/(model.r+chi);
    end
    
    model.numSV(model.iter) = numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end