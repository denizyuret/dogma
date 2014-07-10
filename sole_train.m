function model = sole_train(X,Y,model)
% SOLE_TRAIN Kernel Second Order Label Efficient algorithm
%
%    MODEL = SOLE_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Second Order Label Efficient algorithm. The algorithm will query a
%    label only on certain rounds.
%
%    Additional parameters: 
%    - model.bs governs the sampling rate of the algorithm.
%      Default value is 1.
%
%   References:
%     - Cesa-Bianchi, N., Gentile, C., & Zaniboni, L. (2006).
%       Worst-Case Analysis of Selective Sampling for Linear Classification
%       Journal of Machine Learning Research, 7, (pp. 1205-1230).

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

model.a=1;

if isfield(model,'iter')==0
    model.iter=0;
    model.w=zeros(1,size(X,1));
    model.w2=zeros(1,size(X,1));
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(numel(Y),1);

    model.KbInv=eye(size(X,1))/model.a;
    model.numAskedLabels=zeros(numel(Y),1);
    model.numQueries=0;
end

if isfield(model,'bs')==0
    model.bs=1;
end

for i=1:n
    model.iter=model.iter+1;

    val_f=model.w*X(:,i);

    Yi=Y(i);
    
    KbInv_x=model.KbInv*X(:,i);
    r=X(:,i)'*KbInv_x;
    
    % Include the current sample to predict
    val_f=val_f/(r+1);

    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(model.iter)=val_f;
                
    Z=(rand<model.bs/(abs(val_f)+model.bs));
    model.numQueries=model.numQueries+Z;
   
    if Z==1 && Yi*val_f<=0
        model.S(end+1)=model.iter;
        model.w=model.w+(Yi-val_f)/(1+r)*KbInv_x';
        model.KbInv=model.KbInv-KbInv_x*KbInv_x'/(1+r);
    end
    
    model.numAskedLabels(model.iter)=model.numQueries;
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\tAskedLabels:%5.2f(%d)\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),...
            model.aer(model.iter)*100,model.numQueries/model.iter*100,...
            model.numQueries);
    end
end