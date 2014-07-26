function [predicted_label,margins] = model_predict_gpu(x,model,average)
% MODEL_PREDICT Generic prediction function
%
%    [PREDICTED_LABEL,MARGINS] = MODEL_PREDICT(X,MODEL,AVERAGE) predicts
%    the label of the instances X, using the trained MODEL. If AVERAGE is 1
%    it uses the averaged hyperplane, if present, using the online-to-batch
%    conversion. If AVERAGE is 0 or the averaged hyperplane is not present,
%    it will use the classifier found at the last iteration.
%
%    [PREDICTED_LABEL,MARGINS] = MODEL_PREDICT(X,MODEL) same as above, but
%    setting AVERAGE to 1 by default.
%
%    [PREDICTED_LABEL,MARGINS] = MODEL_PREDICT(K,MODEL,AVERAGE) predicts
%    the label of the instances K, using the trained MODEL. The computed
%    kernel matrix for test data is given as input.
%    If AVERAGE is 1 it uses the averaged hyperplane, if present, using the
%    online-to-batch conversion. If AVERAGE is 0 or the averaged hyperplane
%    is not present, it will use the classifier found at the last
%    iteration.
%
%    [PREDICTED_LABEL,MARGINS] = MODEL_PREDICT(K,MODEL) same as above, but
%    setting AVERAGE to 1 by default.
%
%    Inputs for single kernel:
%    K -  N_training*N_testing Kernel Matricx
%    X -  Dimenstion*N_testing matrix 
%
%    Inputs for multiple kernels:
%    K -  3-D N_training*N_testing matrix*F Kernel Matrices
%    X -  1*F cell, each cell X{f} is a Dimension*N_testing matrix 
%
%    Example:
%        % Define a Gaussian kernel, with scale = 2
%        hp.type='rbf';
%        hp.gamma=1;
%        % Initialize an empy model
%        model=model_init(@compute_kernel,hp);
%        % Trains a kernel Perceptron on the first 5000 samples
%        K = k_perceptron_train(p(:,1:5000),t(1:5000),hp);
%        % Test the last hypothesis on the next 5000 samples
%        preds_last=model_predict(p(:,5001:10000),model);
%        numel(find(preds_last==t(5001:10000)))/numel(preds_last)
%        % Test the average hypothesis on the next 5000 samples
%        preds=model_predict(p(:,5001:10000),model,1);
%        numel(find(preds_av==t(5001:10000)))/numel(preds_av)

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

if nargin<3
    average=1;
end

if isfield(model,'L1') && isfield(model,'L2')
    error('Only kernel models supported\n');
    % two-layers model
    if ~iscell(x) % kernel matrices
        margins=zeros(model.n_cla,size(x,2));
        for i=1:numel(model.L1)
            [dummy, margins_L1]=model_predict(x(:,:,i),model.L1{i},average);
            margins=margins+margins_L1;
        end
    else % cell of features
        margins=zeros(model.n_cla,size(x{1},2));
        for i=1:numel(model.L1)
            [dummy, margins_L1]=model_predict(x{i},model.L1{i},average);
            margins=margins+margins_L1;
        end
    end
elseif isfield(model,'weights')       % MKL model
    error('Only kernel models supported\n');
    if isempty(model.ker)     % input pre-computed kernel matrix
        if isfield(model,'beta2')==0
            average=0;
        end
        if average==0
            margins = full(model.beta(:,model.S))*kbeta(x(model.S, :, :), model.weights')+model.b;
        else
            if size(model.beta2, 1) == numel(model.S)
                margins = full(model.beta2)*kbeta(x(model.S, :, :), model.weights')+model.b2;
           else
                margins = full(model.beta2)*kbeta(x, model.weights')+model.b2;
           end
        end       
    else
        error('MKL model without pre-computed kernels is currently unsupported');
    end
elseif isfield(model,'w')       % primal model
    error('Only kernel models supported\n');
    if isfield(model,'w2')==0
        average=0;
    end
    if average==0
        margins = model.w*x+model.b;
    else
        margins = model.w2*x+model.b2;
    end
else % dual model
    if isempty(model.ker)     % input pre-computed kernel matrix
        error('Only kernel models with support vectors supported\n');
        if isfield(model,'beta2')==0
            average=0;
        end
        if average==0
            %if size(model.beta, 1) == numel(model.S)
                margins = model.beta*x(model.S, :)+model.b;
            %else
            %    margins = model.beta*x+model.b;
            %end
        else
            if size(model.beta2, 1) == numel(model.S)
                margins = model.beta2*x(model.S, :)+model.b;
            else
                margins = model.beta2*x+model.b;
            end
        end
    else      % kernels computed on-the-fly
        hp = model.kerparam;
        if (hp.type ~= 'poly')
           error('Only poly kernels have been implemented.\n');
        end
        if isfield(model,'beta2')==0
            average=0;
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf('Resetting GPU...\n');
        gpudev = gpuDevice(1);

        fprintf('gpu:%g Sending SV to gpu: %g\n', gpudev.FreeMemory/8, numel(model.SV));
        if isfield(model,'X')
          svtr = gpuArray(model.X(:,model.S'));
        else
          svtr = gpuArray(model.SV');
        end
        wait(gpuDevice);

        fprintf('gpu:%g Sending beta to gpu: %g\n', gpudev.FreeMemory/8, numel(model.beta)); 
        if average == 0
          beta = gpuArray(model.beta);
          b = model.b;
        else
          beta = gpuArray(model.beta2);
          b = model.b2;
        end
        wait(gpuDevice);

        % GPU memory 5.0327e+09 according to gpuDevice, 4799MB
        % according to nvidia-smi.  Experiments show it can hold
        % 6.1312e8 doubles and not 6.1313e8.  With multiple arrays
        % capacity drops down to 5.9e8.  Let us take 5e8 as a safe
        % limit.  We have:
        % sv:nd.ns, x:nd.nx, beta:nc.ns margin:nc.nx, k:nsxnk, xx:nd.nk
        % We need space for two sv's (one for the transpose) and one k.
        % No! we need space for one sv and two k's.

        nd = size(x,1);
        nx = size(x,2);
        nc = size(beta,1);
        ns = size(beta,2);
        fprintf('gpu:%g Allocating margins: %g\n', gpudev.FreeMemory/8, nc*nx);
        margins = zeros(nc, nx, 'gpuArray');
        wait(gpuDevice);
        nk = floor((gpudev.FreeMemory/8) / (2*ns+nd));
        assert(nk > 1);
        fprintf('gpu:%g Processing %d chunks of ksize:%dx%d (numel:%g).\n', ...
                gpudev.FreeMemory/8, ceil(nx/nk), ns, nk, (ns*nk));
        wait(gpuDevice);

        for i=1:nk:nx
          j = min(i+nk-1, nx);
          margins(:,i:min(i+nk-1,nx)) = b + beta * (hp.gamma * full(svtr * x(:,i:j)) + hp.coef0).^hp.degree;
          % b + beta * feval(model.ker, sv, x(:,i:min(i+nk-1,nx)), model.kerparam);
          fprintf('.');
        end
        fprintf('\n');
    end
end

% toc();tic();fprintf('Computing labels\n');
if size(margins,1)>1
    % Multiclass
    [~,predicted_label]=max(margins,[],1);
else
    % Binary
    predicted_label=sign(margins);
end

% toc();tic();fprintf('Gathering answers\n');
predicted_label = gather(predicted_label);
margins = gather(margins);
% toc();
