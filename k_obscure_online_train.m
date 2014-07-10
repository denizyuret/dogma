function model = k_obscure_online_train(K, Y, model, options)
% K_OBSCURE_ONLINE_TRAIN  OBSCURE Algorithm stage 1 
%
%    MODEL = K_OBSCURE_ONLINE_TRAIN(K,Y,MODEL) trains an p-norm Multi
%    Kernel classifier using a fast online method, using precomputed kernels.
%     
%    Inputs:
%    K -  3-D N*N*F Kernel Matrices, each kernel K(:, :, i) is a N*N matrix
%    Y -  Training label, N*1 Vector
%
%    Additional parameters:
%    - model.p is 'p' of the p-norm used in the regularization
%      Default value is  1/(1-1/(2*log(number_of_kernels))).
%    - model.T is maximum numer of training epochs for the online stage.
%      The online stage will stop earlier if it converges.
%      Default value is 5.
%    - model.eta 
%      Default value is numbers_of_cue^(-2/q).
%    - model.lambda is the regularization weight.
%      Default value is 1/numel(Y).
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

if isfield(model,'stopCondition')==0
    model.stopCondition = 0;   % threshold on # of updates for terminating
end 

if isfield(model,'step')==0
    model.step = 100*numel(Y); 
end

if isfield(model,'n_cla')==0
    model.n_cla = max(Y); % number of classes
end

if isfield(model,'iter')==0
    model.iter = 0;   
    model.errTot = 0; 
    model.lossTot = 0;

    model.epoch = 0;
    model.time = [];  % training time on each step
    model.test = [];  % iteration when testing happens
end

 if isfield(model,'p')==0
    model.q = 2*log(n_kernel);
    model.p = 1/(1-1/model.q);
else
    model.q = 1/(1-1/model.p);
end

if isfield(model,'eta')==0
    model.eta = n_kernel^(-2/model.q);
end

if isfield(model,'T')==0
    model.T = 5;
end


preds = zeros(model.n_cla, n_kernel);
beta = spalloc(model.n_cla, n, n*model.n_cla);
isSV = zeros(1,n);
weights = zeros(1,n_kernel);
sqnorms = zeros(1,n_kernel)+eps;

val_f = zeros(model.n_cla, 1);

for epoch=1:model.T
    model.epoch = model.epoch+1;
    idx_rand = randperm(n);
    
    n_update=0;
    for i=1:n
        model.iter = model.iter+1;

        idxs_subgrad = idx_rand(i);
        
        if numel(model.S)>0
            K_f = double(K(:, idxs_subgrad, :));    
            preds = beta*K_f;       
            val_f = preds*weights';   
        end

        yi = Y(idxs_subgrad);
        
        margin_true = val_f(yi);
        val_f(yi) = -Inf;
        [margin_pred, yhat] = max(val_f);

        model.errTot = model.errTot+(margin_true<=margin_pred);
        model.lossTot = model.lossTot+max(1-margin_true+margin_pred,0);
        
        % update 
        if margin_true<=margin_pred+1
            beta(yi,idxs_subgrad) = beta(yi,idxs_subgrad)+model.eta;
            beta(yhat,idxs_subgrad) = beta(yhat,idxs_subgrad)-model.eta;

            Kii = double(K(idxs_subgrad,idxs_subgrad, :));
            sqnorms = sqnorms+2*model.eta*(preds(yi, :)-preds(yhat, :))+(2*model.eta^2*Kii(:))';

            isSV(idxs_subgrad) = any(beta(:, idxs_subgrad));
            model.S  = find(isSV);
            n_update = n_update+1;

            norms = sqrt(sqnorms);
            norm_theta = norm(norms+eps,model.q);
            weights = (norms/norm_theta).^(model.q-2)/model.q;
        end

        if mod(model.iter,model.step)==0
            model.test(end+1) = model.iter;
            model.time(end+1) = cputime-timerstart;
            if exist('options') && isfield(options,'eachRound')~=0
                model.beta = beta;
                model.sqnorms = sqnorms;
                model.weights = weights;
                model = feval(options.eachRound, K, Y, model, options);
            end
            timerstart = cputime;
        end
    end

    fprintf('#%.0f(epoch %.0f)\tSV:%5.2f(%d)\tAER:%5.2f\tAEL:%5.2f\tUpdates:%5.2f\n', ...
             ceil(model.iter/1000), epoch, numel(model.S)/n*100, numel(model.S), ...      
             model.errTot/model.iter*100, model.lossTot/model.iter, n_update);

    if n_update<=model.stopCondition || epoch==model.T
        model.test(end+1) = model.iter;  
        model.time(end+1) = cputime-timerstart;
        if exist('options') && isfield(options,'eachRound')~=0
            model.beta = beta;
            model.sqnorms = sqnorms;
            model.weights = weights;
            model = feval(options.eachRound, K, Y, model, options);
        end
        break;
    end
end

model.beta = beta;
model.sqnorms = sqnorms;
model.weights = weights;

if isfield(model, 'obj') && ~isempty(model.obj)
    model.R2 = 2*model.obj(end)/model.lambda;
else
    model.obj = [];
    out = full(model.beta)*kbeta(K, model.weights');
    
    loss=0;
    for i=1:numel(Y)
        margin_true = out(Y(i),i);
        out(Y(i),i) = -Inf;
        margin_pred = max(out(:,i));
        loss = loss+max(1-margin_true+margin_pred,0);
    end  
  
    loss  = loss/numel(Y);
    norms = norm(sqrt(model.sqnorms).*model.weights,model.p)^2;
    model.obj = double(model.lambda*norms/2+loss);  
    model.R2  = double(2*model.obj/model.lambda);
end
