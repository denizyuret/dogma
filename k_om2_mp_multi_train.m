function model = k_om2_mp_multi_train(K, Y, model)
% K_OM2_MP_MULTI_TRAIN  OM-2 Algorithm (Multiple Passes)
%
%    MODEL = K_OM2_MULTI_TRAIN(K,Y,MODEL) trains an p-norm MKL classifier
%    by cyclying on the same training set multiple times using a fast
%    online method.
%    
%    Inputs:
%    K -  3-D N*N*F Kernel Matrices, each kernel K(:, :, i) is a N*N matrix
%    Y -  Training label, 1*N Vector
%
%    Additional parameters:
%    - model.p is 'p' of the p-norm used in the regularization
%      Default value is  1/(1-1/(2*log(numbers_of_cue))).
%    - model.T is maximum numer of training epochs. It will stop earlier if
%      it converges.
%      Default value is 5.
%
%   References:
%     - Jie, L., Orabona, F., Fornoni, M., Caputo, B., and Cesa-Bianchi, N. (2010).
%       "OM-2: An Online Mutli-class Multi-kernel Learning Algorithm". 
%       Proceedings of the 23rd IEEE Conference on Computer Vision and
%       Pattern Recognition - Workshops.

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
%    Contact the authors:  jluo      [at] idiap.ch
%                          francesco [at] orabona.com

timerstart = cputime;

n        = length(Y);   % number of training samples
n_kernel = size(K,3);   % number of kernels

if isfield(model,'stopCondition')==0
    model.stopCondition = 0;   % #. of update threshold
end 

if isfield(model,'step')==0
    model.step = 100*numel(Y); 
end

if isfield(model,'n_cla')==0
    model.n_cla = max(Y); % number of classes
end

if isfield(model,'iter')==0
    model.iter    = 0;   
    model.beta    = spalloc(model.n_cla, n, n*model.n_cla);
    model.errTot  = 0; 
    model.lossTot = 0;
    
    model.S       = [];

    model.epoch = 0;
    model.time  = [];  % training time on each step
    model.test  = [];  % iteration when testing happens
    model.weights = zeros(n_kernel,1);
end

if isfield(model,'p')==0
    model.q = 2*log(n_kernel);
    model.p = 1/(1-1/model.q);
else
    model.q = 1/(1-1/model.p);
end

if isfield(model,'T')==0
    model.T = 5; % maximum number of iterations
end

if isfield(model, 'L1')==0
    model.L1 = cell(n_kernel, 1); 
end

preds   = zeros(model.n_cla, n_kernel);
isSV    = zeros(1,n);
sqnorms = zeros(n_kernel, 1)+eps;

val_f = zeros(model.n_cla, 1);

for epoch=1:model.T
    model.epoch = model.epoch+1;
    idx_rand    = randperm(n);
    
    n_update=0;
    for i=1:n
        model.iter = model.iter+1;

        idxs_subgrad = idx_rand(i);
        
        if numel(model.S)>0
            K_f   = double(K(:, idxs_subgrad, :));    
            preds = model.beta*K_f;
            val_f = preds*model.weights;
        end

        yi = Y(idxs_subgrad);
        
        margin_true = val_f(yi);
        val_f(yi)   = -Inf;
        [margin_pred, yhat] = max(val_f);

        model.errTot  = model.errTot+(margin_true<=margin_pred);
        model.lossTot = model.lossTot+max(1-margin_true+margin_pred,0);
        
        % update 
        if margin_true<=margin_pred+1
            eta = min(1, 1-2*(margin_true-margin_pred)/(2*n_kernel^(2/model.q)));

            model.beta(yi,idxs_subgrad)   = model.beta(yi,idxs_subgrad)+eta;
            model.beta(yhat,idxs_subgrad) = model.beta(yhat,idxs_subgrad)-eta;

            Kii = double(K(idxs_subgrad,idxs_subgrad, :));
            sqnorms = sqnorms+2*eta*(preds(yi, :)-preds(yhat, :))'+(2*eta^2*Kii(:));

            isSV(idxs_subgrad) = any(model.beta(:, idxs_subgrad));
            model.S  = find(isSV);
            n_update = n_update+1;
            
            norms         = sqrt(sqnorms);
            norm_theta    = norm(norms+eps,model.q);
            model.weights = (norms/norm_theta).^(model.q-2)/model.q;
        end

        if mod(model.iter,model.step)==0
            model.test(end+1) = model.iter;
            model.time(end+1) = cputime-timerstart;
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

    fprintf('#%.0f(epoch %.0f)\tSV:%5.2f(%d)\tAER:%5.2f\tAEL:%5.2f\tUpdates:%5.2f\n', ...
             ceil(model.iter/1000), epoch, numel(model.S)/n*100, numel(model.S), ...      
             model.errTot/model.iter*100, model.lossTot/model.iter, n_update);

    if n_update<=model.stopCondition || epoch==model.T
        model.test(end+1) = model.iter;  
        model.time(end+1) = cputime-timerstart;

        if isfield(model,'eachRound')~=0
            if isfield(model, 'outputER')==0
                model.outputER = [];
                model.outputER = feval(model.eachRound, model);
            else
                model.outputER(end+1) = feval(model.eachRound, model);
            end
        end
        break
    end
end
for i=1:numel(model.L1)
    model.L1{i}.S=model.S;
    model.L1{i}.beta=model.beta(:,model.S);
end
