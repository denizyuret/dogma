function model = k_pa_train(X,Y,model)
% K_PA_TRAIN Kernel Passive-Aggressive algorithm
%
%    MODEL = K_PA_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Passive-Aggressive algorithm, PA-I and PA-II variants, using kernels.
%
%    Additional parameters:
%    - model.C is the aggressiveness parameter, used to trade-off the loss
%      on the current sample with the update on the current hyperplane.
%      Default value is 1.
%    - model.update is the used to select the update rule. A value of 1
%      selectes PA-I, 2 selects PA-II.
%      Default value is 1.
%
%   References:
%     - Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., & Singer, Y. (2006).
%       Online Passive-Aggressive Algorithms.
%       Journal of Machine Learning Research 7(Mar), (pp. 551-585).

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
end

if isfield(model,'update')==0
    model.update=1; %default update using PA-I
end

if isfield(model,'C')==0
    model.C=1;
end

for i=1:n
    model.iter=model.iter+1;
        
    if numel(model.S)>0
        if isempty(model.ker)
            K_f=X(i,model.S);
        else
            K_f=feval(model.ker,model.SV,X(:,i),model.kerparam);
        end
        val_f=model.beta*K_f;
    else
        val_f=0;
    end

    model.errTot=model.errTot+(sign(val_f)~=Y(i));
    model.aer(model.iter)=model.errTot/model.iter;

    model.pred(model.iter)=val_f;
    
    if Y(i)*val_f<=1
        if isempty(model.ker)
            Kii=X(i,i);
        else
            Kii=feval(model.ker,X(:,i),X(:,i),model.kerparam);
            model.SV(:,end+1)=X(:,i);
        end
        if model.update==1
            new_beta=min((1-val_f*Y(i))/Kii,model.C);
        else
            new_beta=(1-val_f*Y(i))/(Kii+1/(2*model.C));
        end
        model.beta(end+1)=Y(i)*new_beta;
        model.S(end+1)=model.iter;
        
        model.beta2(end+1)=0;
    end

    model.beta2=model.beta2+model.beta;
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end
