function model = k_ufomkl_train(K, Y, model, options)
% K_UFOMKL_TRAIN Ultra Fast Optimization for Multi Kernel Learning
%
%    MODEL = K_UFOMKL_TRAIN(K,Y,MODEL) trains a sparse Multi Kernel binary
%    classifier using Ultra Fast Optimization algorithm, using precomputed
%    kernels. The loss function is the hinge loss.
%
%    Inputs:
%    K -  3-D N*N*F Kernel Matrices, each kernel K(:, :, i) is a N*N matrix
%    Y -  Training label, N*1 Vector 
%
%    Additional parameters:
%    - model.alpha is the weight of the group norm (2,1) term. It regulates
%      the sparsity of the solution.
%      Default value is 0.01.
%    - model.T is numer of training epochs for the batch stage.
%      Default value is 5.
%    - model.lambda is the regularization weight.
%      Default value is 1/numel(Y).
%
%   References:
%     - Orabona, F., Jie, L. (2011).
%       Ultra-Fast Optimization Algorithm for Sparse Multi Kernel Learning.
%       Proceedings of the 28th International Conference on Machine Learning.

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
  model.lambda = 1/numel(Y);
end

if isfield(model,'step')==0
    model.step = 100*numel(Y);
end

if isfield(model,'alpha')==0
    model.alpha = .01;
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
    model.sparse = [];
end

beta = spalloc(1, n, n);
predmat = zeros(n, n_kernel);
weights = zeros(1, n_kernel);
sqnorms = zeros(1, n_kernel)+eps;
model.time = [];   % training time on each step

model.q = 2*log(n_kernel);
model.p = 1/(1-1/model.q);

if isfield(model,'T')==0
    model.T = 5;
end

for epoch = 1:model.T
    model.epoch = model.epoch+1;
    idx_rand = randperm(n);
    
    model.errTot = 0;
    model.lossTot = 0;

    n_update = 0;
    for i = 1:n
        model.iter = model.iter+1;

        idxs_subgrad = idx_rand(i);
        
        preds = predmat(idxs_subgrad,:);
        val_f = preds(:, :)*weights';

        yhat = sign(val_f); 
        yi = Y(idxs_subgrad);

        loss = max(1-yi*val_f, 0);

        model.errTot = model.errTot  + (yhat~=yi);
        model.lossTot = model.lossTot + loss;
        
        lr = model.lambda*model.iter;
        
        if loss>0
            Kii = double(K(idxs_subgrad,idxs_subgrad, :));

            beta(idxs_subgrad) = beta(idxs_subgrad)+yi;
            sqnorms = sqnorms + 2*yi*preds + Kii(:)';

            n_update = n_update+1;
            
            % update predmat
            predmat = predmat + squeeze(yi*K(idxs_subgrad,:,:));
        
            norms = sqrt(sqnorms);
            trunc_norms = max(norms-model.iter*model.alpha,0);
            norm_trunc_theta = norm(trunc_norms+eps,model.q);
            weights = (trunc_norms./(norms+eps)).*((trunc_norms/norm_trunc_theta).^(model.q-2))/lr;
        else
            trunc_norms = max(norms-model.iter*model.alpha,0);
            norm_trunc_theta = norm(trunc_norms+eps,model.q);
            weights = (trunc_norms./(norms+eps)).*((trunc_norms/norm_trunc_theta).^(model.q-2))/lr;
        end

        if mod(model.iter+model.inititer,model.step)==0
            model.test(end+1) = model.iter+model.inititer;
            model.time(end+1) = cputime-timerstart;
            if exist('options') && isfield(options,'eachRound')~=0
                model.beta = beta;
                model.sqnorms = sqnorms;
                model.weights = weights;
                model = feval(options.eachRound, K, Y, model, options);
            end
            model.sparse(end+1) = numel(find(weights==0));
	        timerstart = cputime;
        end
    end

    fprintf('#%.0f(epoch %.0f)\tAER:%5.2f\tAEL:%5.2f\tUpdates:%.0f\n', ...
             ceil(model.iter/1000), epoch, model.errTot/n*100, model.lossTot/n, n_update);

    if epoch==model.T
        model.test(end+1) = model.iter+model.inititer;  
        model.time(end+1) = cputime-timerstart;
        if exist('options') && isfield(options,'eachRound')~=0
            model.beta = beta;
            model.sqnorms = sqnorms;
            model.weights = weights;
            model = feval(options.eachRound, K, Y, model, options);
        end
        model.sparse(end+1) = numel(find(weights==0));
       	timerstart = cputime;
    end
end

model.beta = beta;
model.sqnorms = sqnorms;
model.weights = weights;
model.S = find(beta);