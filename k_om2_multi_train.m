function model = k_om2_multi_train(X, Y, model)
% K_OM2_MULTI_TRAIN  OM-2 Algorithm
%
%    MODEL = K_OM2_MULTI_TRAIN(X,Y,MODEL) trains an p-norm Multi-kernel
%    Multi-class classifier, according to the OM-2 algorithm.
%     
%    MODEL = K_OM2_MULTI_TRAIN(K,Y,MODEL) trains an p-norm Multi-kernel
%    Multi-class classifier, according to the OM-2 algorithm, and using
%    precomputed kernels.
%    
%    Inputs:
%    K -  3-D N*N*F Kernel Matrices, each kernel K(:, :, i) is a N*N matrix
%    X -  1*F cell, each cell X{f} is a D*N matrix 
%    Y -  Training label, 1*N Vector
%
%    Additional parameters:
%    - model.p is 'p' of the p-norm used in the regularization
%      Default value is  1/(1-1/(2*log(numbers_of_cue))).
%
%   References:
%     - Jie, L., Orabona, F., Fornoni, M., Caputo, B., and Cesa-Bianchi, N. (2010).
%       OM-2: An Online Mutli-class Multi-kernel Learning Algorithm.
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

n = length(Y);   % number of training sample
if iscell(X)
    n_kernel = numel(X); % number of features
else
    n_kernel = size(X,3);  
end

if isfield(model,'stopCondition')==0
    model.stopCondition = 0;   % #. of update threshold
end

if isfield(model,'n_cla')==0
    model.n_cla = max(Y);
end

if isfield(model,'p')==0
    model.q = 2*log(n_kernel);
    model.p = 1/(1-1/model.q);
else
    model.q = 1/(1-1/model.p);
end

if isfield(model,'iter')==0
    model.iter    = 0;
    model.beta    = [];
    model.errTot  = 0;
    model.numSV   = zeros(numel(Y),1);
    model.aer     = zeros(numel(Y),1);

    model.S       = [];

    model.weights = ones(n_kernel, 1);
    model.time    = [];   % training time on each step
    model.test    = [];   % iteration when testing happens
end

sqnorms = zeros(n_kernel, 1)+eps;
Kii     = zeros(n_kernel, 1);

for i=1:n
    model.iter = model.iter+1;

    if numel(model.S)>0
        K_f=zeros(numel(model.S), n_kernel);
        if ~iscell(X)
            K_f = double(X(model.S, i, :));
        else
            for j=1:n_kernel
                K_f(:, j) = feval(model.L1{j}.ker, model.L1{j}.SV, X{j}(:,i), model.L1{j}.kerparam);
            end
        end
        preds = model.beta*K_f;
        val_f = preds*model.weights;
    else
        preds = zeros(model.n_cla, n_kernel);
        val_f = preds*model.weights;
    end

    yi = Y(i);

    margin_true = val_f(yi); 
    val_f(yi)   = -Inf;
    [margin_pred,yhat] = max(val_f);

    model.errTot          = model.errTot+(margin_true<=margin_pred);
    model.aer(model.iter) = model.errTot/model.iter;

    % update
    if margin_true<=margin_pred+1
        eta = min(1, 1-2*(margin_true-margin_pred)/(2*n_kernel^(2/model.q)));
        if ~iscell(X)
            Kii = double(X(i, i, :));
        else
            for j=1:n_kernel
                Kii(j)=feval(model.L1{j}.ker,X{j}(:,i),X{j}(:,i),model.L1{j}.kerparam);
                model.L1{j}.SV(:,end+1)=X{j}(:,i);
            end
        end

        sqnorms = sqnorms+2*eta*(preds(yi, :)-preds(yhat, :))'+(2*eta^2*Kii(:));

        model.beta(:, end+1)  = spalloc(model.n_cla, 1, 2);
        model.beta(yi, end)   = eta;
        model.beta(yhat, end) = -eta;

        model.S(end+1) = model.iter;

        norms         = sqrt(sqnorms);
        norm_theta    = norm(norms, model.q);
        model.weights = (norms/norm_theta).^(model.q-2)/model.q;
    end

    model.numSV(model.iter) = numel(model.S);

    if mod(i,model.step)==0 || i==n
        model.test(end+1) = model.iter;
        model.time(end+1) = cputime-timerstart;
        timerstart = cputime;
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
    end
end
for i=1:numel(model.L1)
    model.L1{i}.S=model.S;
    model.L1{i}.beta=model.beta;
end