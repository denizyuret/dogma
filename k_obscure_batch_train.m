function model = k_obscure_batch_train(K, Y, model, options)
% K_OBSCURE_BATCH_TRAIN OBSCURE Algorithm stage 2 
%
%    MODEL = K_OBSCURE_BATCH_TRAIN(K,Y,MODEL) trains an p-norm Multi Kernel
%    classifier using a stochastic subgradient method, using precomputed
%    kernels.
%
%    Inputs:
%    K -  3-D N*N*F Kernel Matrices, each kernel K(:, :, i) is a N*N matrix
%    Y -  Training label, N*1 Vector 
%
%    Additional parameters:
%    - model.p is 'p' of the p-norm used in the regularization
%      Default value is  1/(1-1/(2*log(number_of_kernels))).
%    - model.T is numer of training epochs for the batch stage.
%      Default value is 5.
%    - model.lambda is the regularization weight.
%      Default value is 1/numel(Y).
%    - model.R2 is upperbound on the squared norm of the optimal solution.
%      Default value is 2/model.lambda, if a bigger finite value is
%      supplied, the default value is used instead.
%      If the value 'inf' is used, the proximal regularization and the
%      projection step are not used.
%
%   References:
%     - Orabona, F., Jie, L., and Caputo, B. (2010).
%       Online-Batch Strongly Convex Multi Kernel Learning.
%       Proceedings of the 23rd IEEE Conference on Computer Vision and
%       Pattern Recognition.

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
%    Contact the authors: francesco [at] orabona.com
%                         jluo      [at] idiap.ch

timerstart = cputime;

n = length(Y);   % number of training samples
n_kernel = size(K,3);   % number of kernels

if isfield(model,'lambda')==0
  if isfield(model, 'C')==0  
     model.lambda = 1/numel(Y);
  else
     model.lambda = 1/(numel(Y)*model.C);
  end
end

if isfield(model,'R2')==0
    model.R2 = 2/model.lambda;
elseif model.R2~=inf
    model.R2 = min(2/model.lambda,model.R2);
end

if isfield(model,'step')==0
    model.step = 100*numel(Y);
end

if isfield(model,'n_cla')==0
    model.n_cla = max(Y);
end

if isfield(model,'iter')==0
    model.iter = 0;
    model.aer = [];

    model.epoch = 0;
    model.sum_tau = 0;
end

if isfield(model, 'test')==1
    model.inititer = model.test(end);
else
    model.inititer = 0;
    model.test = [];
end

beta = spalloc(n, model.n_cla, n*model.n_cla);
predmat = zeros(model.n_cla, n, n_kernel);
isSV = zeros(1,n);
if isfield(model,'beta')==1
    beta(model.S,:)=model.beta(:, model.S)';
    weights = model.weights;
    sqnorms = model.sqnorms;
    predmat = reshape(model.beta(:, model.S)*double(K(model.S, :, :)), size(predmat));
    for i=1:size(beta,1)
        isSV(i)=any(beta(i,:));
    end
else
    weights = zeros(1,n_kernel);
    sqnorms = zeros(1,n_kernel)+eps;
    model.time = [];   % training time on each step
end

if isfield(model,'p')==0
    model.q = 2*log(n_kernel);
    model.p = 1/(1-1/model.q);
else
    model.q = 1/(1-1/model.p);
end
qR=sqrt(model.R2)*model.q;

if isfield(model,'T')==0
    model.T = 5;
end

cum_fact_predmat = ones(model.n_cla, 1);
cum_fact_beta = ones(model.n_cla,1);
norm_theta = norm(sqrt(sqnorms)+eps,model.q);

for epoch=1:model.T
    model.epoch=model.epoch+1;
    idx_rand=randperm(n);
    
    model.errTot = 0;
    model.lossTot = 0;

    n_update=0;
    for i=1:n
        model.iter = model.iter+1;

        idxs_subgrad = idx_rand(i);
        
        preds = predmat(:,idxs_subgrad,:);
        val_f = preds(:, :)*weights';

        yi = Y(idxs_subgrad);
        
        val_f = val_f.*cum_fact_predmat;
        
        margin_true = val_f(yi);
        val_f(yi) = -Inf;
        [margin_pred, yhat] = max(val_f);

        model.errTot = model.errTot+(margin_true<=margin_pred);
        model.lossTot = model.lossTot+max(1-margin_true+margin_pred,0);
        
        if margin_true<1+margin_pred
            a = model.iter*model.lambda+model.sum_tau;
            tau = (-a+sqrt(a^2+model.q*(n_kernel^(1/model.q)*sqrt(2)+model.lambda/model.q*norm_theta)^2/model.R2))/2;
            model.sum_tau = model.sum_tau+tau;
            eta = model.q/(model.lambda*model.iter+model.sum_tau);

            % Gradient descent step
            fact = 1-eta*model.lambda/model.q;

            sqnorms = sqnorms*(fact^2);        
            cum_fact_predmat = cum_fact_predmat*fact;

            Kii = double(K(idxs_subgrad,idxs_subgrad, :));
            sqnorms = sqnorms+2*eta*(preds(yi, :)*cum_fact_predmat(yi)-preds(yhat, :)*cum_fact_predmat(yhat))+(2*eta^2*Kii(:))';
            
            predmat(yi,:,:) = predmat(yi,:,:)*cum_fact_predmat(yi);
            cum_fact_predmat(yi) = 1;
            predmat(yhat,:,:) = predmat(yhat,:,:)*cum_fact_predmat(yhat);
            cum_fact_predmat(yhat) = 1;
            
            beta(:,yi) = beta(:,yi)*cum_fact_beta(yi);
            cum_fact_beta(yi) = 1;
            beta(:,yhat) = beta(:,yhat)*cum_fact_beta(yhat);
            cum_fact_beta(yhat) = 1;
            
            beta(idxs_subgrad,yi) = beta(idxs_subgrad,yi)+eta;
            beta(idxs_subgrad,yhat) = beta(idxs_subgrad,yhat)-eta;
            
            n_update = n_update+1;
            
            % update predmat
            tmp = K(idxs_subgrad,:,:)*eta;
            predmat(yi,:,:) = predmat(yi, :, :)+tmp;
            predmat(yhat,:,:) = predmat(yhat,:,:)-tmp;
            
            norms = sqrt(sqnorms);
            norm_theta = norm(norms+eps,model.q);
            weights = (norms/norm_theta).^(model.q-2)/model.q;
        else
            a = model.iter*model.lambda+model.sum_tau;
            tau = (-a+sqrt(a^2+model.q*(model.lambda/model.q*norm_theta)^2/model.R2))/2;
            model.sum_tau = model.sum_tau+tau;
            eta = model.q/(model.lambda*model.iter+model.sum_tau);

            % Gradient descent step
            fact = 1-eta*model.lambda/model.q;

            sqnorms = sqnorms*(fact^2);
            
            norm_theta = norm_theta*fact;
            % weights remain the same
            %weights=(norms/norm_theta).^(model.q-2)/model.q;
            
            cum_fact_predmat = cum_fact_predmat*fact;
            cum_fact_beta = cum_fact_beta*fact;
        end

        % Projection step & w_{t+1}=nabla*(theta_{t+1})
        %norms=sqrt(sqnorms);
        %norm_theta = norm(norms+eps,model.q);
        %weights=(norms/norm_theta).^(model.q-2)/model.q;
        
        if qR<norm_theta
            % weights does not change by the scaling
            fact = qR/norm_theta;
            sqnorms = sqnorms*fact^2;
            cum_fact_predmat = cum_fact_predmat*fact;
            cum_fact_beta = cum_fact_beta*fact;
      
            norm_theta = norm_theta*fact;
        end
        if mod(model.iter+model.inititer,model.step)==0
            model.test(end+1) = model.iter+model.inititer;
            model.time(end+1) = cputime-timerstart;
            if exist('options') && isfield(options,'eachRound')~=0
                for j=1:model.n_cla
                    beta(:,j) = beta(:,j)*cum_fact_beta(j);
                    cum_fact_beta(j) = 1;
                end
                model.beta = beta';
                model.sqnorms = sqnorms;
                model.weights = weights;
                model = feval(options.eachRound, K, Y, model, options);
            end
	        timerstart = cputime;
        end
    end

    fprintf('#%.0f(epoch %.0f)\tAER:%5.2f\tAEL:%5.2f\tUpdates:%.0f\tR2:%d\n', ...
             ceil(model.iter/1000), epoch, model.errTot/n*100, model.lossTot/n, n_update, model.R2);

    if epoch==model.T
        model.test(end+1) = model.iter+model.inititer;  
        model.time(end+1) = cputime-timerstart;
        if exist('options') && isfield(options,'eachRound')~=0
           for j=1:model.n_cla
                beta(:,j) = beta(:,j)*cum_fact_beta(j);
                cum_fact_beta(j) = 1;
           end
           model.beta = beta';
           model.sqnorms = sqnorms;
           model.weights = weights;
           model = feval(options.eachRound, K, Y, model, options);
        end
       	timerstart = cputime;
    end
end

for j=1:model.n_cla
    beta(:,j) = beta(:,j)*cum_fact_beta(j);
    cum_fact_beta(j) = 1;
end
model.beta = beta';
model.sqnorms = sqnorms;
model.weights = weights;
for i=1:size(beta,1)
    isSV(i) = any(beta(i,:));
end
model.S = find(isSV);
