function model = sop_train(X,Y,model)
% SOP_TRAIN Second-order Perceptron algorithm
%
%    MODEL = SOP_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Second-order Perceptron algorithm.
%
%    Additional parameters:
%    - model.a is the aggressiveness parameter, used to trade-off the loss
%      and the regularization.
%      Default value is 1.
%
%   References:
%     - Cesa-Bianchi, N., Conconi, A., & Gentile, C. (2005).
%       A Second Order Perceptron Algorithm.
%       SIAM J. COMPUT. 34(3), (pp. 640-668).

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

n = length(Y);   % number of training samples

if isfield(model,'a')==0
    model.a=1;
end

if isfield(model,'iter')==0
    model.iter=0;
    model.w=zeros(1,size(X,1));
    model.w2=zeros(1,size(X,1));
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(numel(Y),1);

    model.KbInv=model.a*eye(size(X,1));
end

for i=1:n
    model.iter=model.iter+1;
        
    val_f=model.w*X(:,i);

    Yi=Y(i);
    
    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(model.iter)=val_f;

    if Yi*val_f<=0
        KbInv_x=model.KbInv*X(:,i);
        v=X(:,i)'*KbInv_x;
        
        model.w=model.w+(Yi-val_f)/(1+v)*KbInv_x';
        model.KbInv=model.KbInv-KbInv_x*KbInv_x'/(1+v);
        
        model.S(end+1)=model.iter;
        %model.SV(:,end+1)=X(:,i);
    end

    model.w2=model.w2+model.w;
    
    model.numSV(model.iter)=numel(model.S);

    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end
