function model = k_pegasos_train(X,Y,model)
% K_PEGASOS_TRAIN Kernel Pegasos Algorithm
%
%    MODEL = K_PEGASOS_TRAIN(X,Y,MODEL) trains an classifier according to
%    Pegasos algorithm, using kernels.
%
%    Additional parameters:
%    - model.k is the number of samples used to estimate the gradient at
%      each step.
%      Default value is 1.
%    - model.T is the numer of epochs, as a fraction of the number of
%      training points.
%      Default value is 5.
%    - model.lambda is the regularization weight.
%      Default value is 1/numel(Y).
%
%    Note that the projection step is missing in this implementation. 
%
%   References:
%     - Shalev-Shwartz, S., Singer, Y., & Srebro, N. (2007)
%       Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
%       Proceedings of the 24th International Conference on Machine
%       Learning.
%     - Shalev-Shwartz, S., & Srebro, N. (2008)
%       SVM Optimization: Inverse Dependence on Training Set Size.
%       Proceedings of the 25th International Conference on Machine
%       Learning.

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
    model.beta=spalloc(1,n,n);
    model.errTot=0;
    model.aer=[];
    
    model.num_ker_eval=0;
    model.X=[];
    model.Y=[];
    model.epoch=0;
end

if isfield(model,'k')==0
    model.k=1;
end

if isfield(model,'T')==0
    model.T=10;
end

if isfield(model,'lambda')==0
    model.lambda=1/numel(Y);
end

for epoch=1:model.T
    model.epoch=model.epoch+1;
    idx_rand=randperm(n);
    
    for i=1:model.k:n
        model.iter=model.iter+1;

        idxs_for_subgrad=idx_rand(i:i+model.k-1);
        
        if numel(model.S)>0
            if isempty(model.ker)
                K_f=X(model.S,idxs_for_subgrad);
            else
                K_f=feval(model.ker,model.SV,X(:,idxs_for_subgrad),model.kerparam);
            end
            val_f=model.beta(model.S)*K_f;
        else
            val_f=zeros(1,model.k);
        end

        tmpY=Y(idxs_for_subgrad);
        
        model.errTot=model.errTot+sum(sign(val_f)~=tmpY);
        model.aer(model.iter)=model.errTot/(model.iter*model.k);

        eta=1/(model.lambda*model.iter);
        model.beta=model.beta*(1-model.lambda*eta);
        
        idx_to_update=find(val_f.*tmpY<1);
        if numel(idx_to_update)>0
            model.beta(idxs_for_subgrad(idx_to_update))=...
                model.beta(idxs_for_subgrad(idx_to_update))+eta*tmpY(idx_to_update)/model.k;
            
            model.S=find(model.beta);
            if ~isempty(model.ker)
                model.SV=X(:,model.S);
            end
        end

        if mod(model.iter,model.step)==0
            fprintf('#%.0f(epoch %.0f)\tSV:%5.2f(%d)\tAER:%5.2f\n', ...
                ceil(model.iter/1000),epoch,numel(model.S)/n*100,numel(model.S),model.aer(end)*100);
        end
    end
end
model.X=X;
model.S=find(model.beta);
model.SV=X(:,model.S);
model.beta=full(model.beta(model.S));