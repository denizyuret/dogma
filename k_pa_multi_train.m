function model = k_pa_multi_train(X,Y,model)
% K_PA_MULTI_TRAIN Kernel Passive Aggressive multiclass algorithm
%
%    MODEL = K_PA_MULTI_TRAIN(X,Y,MODEL) trains a multiclass classifier
%    according to the Passive-Aggressive algorithm, PA-I and PA-II
%    variants, using kernels.
%
%    MODEL = K_PA_MULTI_TRAIN(K,Y,MODEL) trains a multiclass classifier
%    according to the Passive-Aggressive algorithm, PA-I and PA-II
%    variants, using kernels. The kernel matrix is given as input.
%
%    If the maximum number of Support Vectors is inf, the algorithm also
%    calculates an averaged solution.
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

if isfield(model,'n_cla')==0
    model.n_cla=max(Y);
end

if isfield(model,'iter')==0
    model.iter=0;
    model.time=[];
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
    model.update=1; %default update using PA-I
end

if isfield(model,'C')==0
    model.C=1;
end

if isfield(model,'maxSV')==0
    model.maxSV=inf;
end

timerstart = cputime;
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
        val_f=zeros(model.n_cla,1);
    end
    
    Yi=Y(i);
    
    tmp=val_f; tmp(Yi)=-inf;
    [mx_val,idx_mx_val]=max(tmp);

    model.errTot=model.errTot+(val_f(Yi)<=mx_val);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(:,model.iter)=val_f;
        
    if val_f(Yi)<1+mx_val
        model.S(end+1)=model.iter;

        if isempty(model.ker)
            norm_x_square=2*X(i,i);
        else
            norm_x_square=2*feval(model.ker,X(:,i),X(:,i),model.kerparam);
            model.SV(:,end+1)=X(:,i);
        end

        if model.update==1
            new_beta=min((1-(val_f(Yi)-mx_val))/norm_x_square,model.C);
        else
            new_beta=(1-(val_f(Yi)-mx_val))/(norm_x_square+1/(2*model.C));
        end
        model.beta(:,end+1)=spalloc(1,model.n_cla,2);
        model.beta(Yi,end)=new_beta;
        model.beta(idx_mx_val,end)=-new_beta;
        
        if model.maxSV==inf
            model.beta2(:,end+1)=spalloc(model.n_cla,1,2);
        end
        
        if size(model.SV,2)>model.maxSV
            if isfield(model,'randDiscard')
                mn_idx=ceil(model.maxSV*rand);
            else
                [mn,mn_idx]=min(max(model.beta,[],2));
            end
            model.beta(:,mn_idx)=[];
            model.SV(:,mn_idx)=[];
            model.S(mn_idx)=[];
        end
    end

    if model.maxSV==inf
        model.beta2=model.beta2+model.beta;
    end
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0 || i==n
	  model.time(end+1)=cputime-timerstart;
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);

      if isfield(model,'eachRound')~=0
        if isfield(model, 'outputER')==0
          model.outputER = [];
          model.outputER = feval(model.eachRound, model);
        else
          model.outputER(end+1) = feval(model.eachRound, model);
        end
      end
	  timerstart = cputime;
    end
end
