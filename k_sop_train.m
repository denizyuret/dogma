function model = k_sop_train(X,Y,model)
% K_SOP_TRAIN Kernel Second-order Perceptron algorithm
%
%    MODEL = K_SOP_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Second-order Perceptron algorithm, using kernels.
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

if isfield(model,'iter')==0
    model.iter=0;
    model.beta=[];
    model.beta2=[];
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(numel(Y),1);

    model.KbInv=[];
    model.Y_S=[];
end

if isfield(model,'a')==0
    model.a=1;
end

for i=1:n
    model.iter=model.iter+1;
        
    if numel(model.S)>0
        if isempty(model.ker)
            K_f=X(model.S,i);
        else
            K_f=feval(model.ker,model.SV,X(:,i),model.kerparam);
        end
        val_f=model.beta*K_f;
    else
        val_f=0;
    end

    Yi=Y(i);
    
    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(model.iter)=val_f;

    if Yi*val_f<=0
        dimS=numel(model.S);
        if isempty(model.ker)
            Kii=X(i,i);
        else
            Kii=feval(model.ker,X(:,i),X(:,i),model.kerparam);
            model.SV(:,end+1)=X(:,i);
        end
        if dimS>0
            % incremental update of the inverse matrix
            v=model.KbInv*K_f;
            d=Kii+model.a-K_f'*v;
            model.KbInv=[model.KbInv, zeros(dimS,1);zeros(1,dimS+1)];
            model.KbInv=model.KbInv+[v; -1]*[v; -1]'/d;
        else    
            model.KbInv=full(1/(Kii+model.a));
        end
        
        model.S(end+1)=model.iter;
        model.Y_S(end+1)=Yi;
        model.beta=model.Y_S*model.KbInv;
        
        model.beta2(end+1)=0;
    end

    model.beta2=model.beta2+model.beta;
    
    model.numSV(model.iter)=numel(model.S);

    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end
