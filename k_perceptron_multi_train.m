function model = k_perceptron_multi_train(X,Y,model)
% K_PERCEPTRON_MULTI_TRAIN Kernel Perceptron multiclass algorithm
%
%    MODEL = K_PERCEPTRON_MULTI_TRAIN(X,Y,MODEL) trains a multiclass
%    classifier according to the Perceptron algorithm, using kernels.
%
%    MODEL = K_PERCEPTRON_MULTI_TRAIN(K,Y,MODEL) trains a multiclass
%    classifier according to the Perceptron algorithm, using kernels. The
%    kernel matrix is given as input.
%
%    If the maximum number of Support Vectors is inf, the algorithm also
%    calculates an averaged solution.
%
%    Additional parameters: 
%    - model.maxSV is the maximum number of Support Vectors. When the
%      algorithm reaches that quantity it starts discarding random vectors,
%      according to the Random Budget Perceptron algorithm.
%      Default value is inf.
%
%   References:
%     - Crammer, K., & Singer Y. (2003).
%       Ultraconservative Online Algorithms for Multiclass Problems.
%       Journal of Machine Learning Research 3, (pp. 951-991).

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

if isfield(model,'n_cla')==0
    model.n_cla=max(Y);
end

if isfield(model,'iter')==0
    model.iter=0;
    model.beta=[];
    model.beta2=[];
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(model.n_cla,numel(Y));
else
    assert(isfield(model,'ker'), 'Cannot continue training using a Kernel matrix as input.');
end

if isfield(model,'update')==0
    model.update=1; % max-score
end

if isfield(model,'maxSV')==0
    model.maxSV=inf;
end

for i=1:n
    model.iter=model.iter+1;
    
    if numel(model.S)>0
        if isempty(model.ker)
            K_x=X(model.S,i);
        else
            K_x=feval(model.ker,model.SV,X(:,i),model.kerparam);
        end
        val_f=model.beta*K_x;
    else
        val_f=zeros(1,model.n_cla);
    end
    
    Yi=Y(i);
    
    tmp=val_f; tmp(Yi)=-inf;
    [mx_val,idx_mx_val]=max(tmp);

    model.errTot=model.errTot+(val_f(Yi)<=mx_val);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(:,model.iter)=val_f;
    
    if val_f(Yi)<=mx_val
        model.S(end+1)=model.iter;
        if ~isempty(model.ker)
            model.SV(:,end+1)=X(:,i);
        end

        model.beta(:,end+1)=zeros(model.n_cla,1);
        if model.update==1
            % max-score
            model.beta(Yi,end)=1;
            model.beta(idx_mx_val,end)=-1;
        else
            % uniform
            model.beta(:,end)=-1/(model.n_cla-1);
            model.beta(Yi,end)=1;
        end
        
        if model.maxSV==inf
            model.beta2(:,end+1)=zeros(model.n_cla,1);
        end
        
        if numel(model.S)>model.maxSV
            mn_idx=ceil(model.maxSV*rand);
            model.beta(:,mn_idx)=[];
            if isfield(model,'ker')
                model.SV(:,mn_idx)=[];
            end
            model.S(mn_idx)=[];
        end
    end

    if model.maxSV==inf
        model.beta2=model.beta2+model.beta;
    end
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end
