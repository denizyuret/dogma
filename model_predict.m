function [predicted_label,margins] = model_predict(x,model,average)
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
        if isfield(model,'beta2')==0
            average=0;
        end

        nt = size(x,2);
        max_num_el=100*1024^2/8; %50 Mega of memory as maximum size for K
        step=ceil(max_num_el/size(model.beta,1));
        for i=1:step:nt
            if isfield(model,'X')
                K = feval(model.ker,model.X(:,model.S),x(:,i:min(i+step-1,nt)),model.kerparam);
            else
                K = feval(model.ker,model.SV,x(:,i:min(i+step-1,nt)),model.kerparam);
            end
            if average==0
                margins(:,i:min(i+step-1,nt)) = model.beta*K+model.b;
            else
                margins(:,i:min(i+step-1,nt)) = model.beta2*K+model.b2;
            end
        end
    end
end

if size(margins,1)>1
    % Multiclass
    [tmp,predicted_label]=max(margins,[],1);
else
    % Binary
    predicted_label=sign(margins);
end