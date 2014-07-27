function model = k_perceptron_multi_train_gpu_dbg(X,Y,model)
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
%
%    GPU extension: Deniz Yuret, July 27, 2014

% Make sure we have a poly kernel model
assert(isfield(model,'ker') && ~isempty(model.ker) && isfield(model,'kerparam'), 'Only kernel models supported.\n');
hp = model.kerparam;
assert(strcmp(hp.type,'poly'), 'Only poly kernel models supported.\n');

% Get the size of the problem
nd = size(X, 1);
nx = size(X, 2);
nc = max(Y);
if isempty(model.SV)
  ns = 0;
else
  ns = size(model.SV, 2);
end
if isfield(model,'n_cla')==0
  model.n_cla=nc;
end
assert(nc == model.n_cla);
assert(nx == numel(Y));
fprintf('nd=%d nx=%d nc=%d ns=%d\nResetting gpu.\n', nd, nx, nc, ns);
tic;gpu=gpuDevice(1);toc;

% Setup model on GPU
tic;
if isfield(model,'iter')==0
  assert(ns == 0);
  fprintf('Initializing new model.\n');
  model.iter=0;
  model.beta=zeros(nc, 0, 'gpuArray');
  model.beta2=zeros(nc, 0, 'gpuArray');
  model.S = zeros(1, 0, 'gpuArray');
  SVtr = { zeros(0, nd, 'gpuArray') };
  model.errTot=0;
  model.numSV=zeros(nx,1);
  model.aer=zeros(nx,1);
  model.pred=zeros(nc,nx);
else
  assert(ns == size(model.SV, 2));
  assert(isfield(model,'ker'), ['Cannot continue training using a Kernel matrix as input.']);
  fprintf('g=%g Loading model to gpu...\n', gpu.FreeMemory/8);
  model.beta = gpuArray(model.beta);
  model.beta2 = gpuArray(model.beta2);
  model.S = gpuArray(model.S);
  SVtr = { gpuArray(model.SV') zeros(0, nd, 'gpuArray') };
end
wait(gpuDevice);
toc;

assert(~isfield(model,'update') || model.update == 1, 'Only model.update==1 supported.');
assert(~isfield(model,'maxSV') || model.maxSV == inf, 'Only model.maxSV==inf supported.');

% Stupid matlab copies on write, so we need to keep sv in small blocks
sv_block_size = floor(1e8/nd);

% Batchsize is for X, sv_block_size was for SV
if isfield(model,'batchsize')==0
  model.batchsize=1000;
end
batchsize_warning = 0;
fprintf('g=%g Using X batchsize=%d, SV blocksize=%d\n', gpu.FreeMemory/8, model.batchsize, sv_block_size);

if isfield(model,'epochs')==0
  model.epochs = 1;
end

model.ttimes=0;
model.ttimes2=0;
model.t=zeros(1,30);

tic;
for epoch=1:model.epochs
  % TODO: We should shuffle here?  it would make S=iter indices meaningless.  
  % Check dogma models supporting multi-epoch.

  i = 1; 		% will process X(:,i:j)
  while (i <= nx)       % 26986us/iter to process X(:,i:i+500)
    model.iter=model.iter+1;

    % Compute the real batchsize here based on memory
    assert(ns == size(model.beta,2));
    nk = floor((gpu.FreeMemory/8) / (ns+sv_block_size+2*nd+5*nc+10));
    nk = min(nk, model.batchsize);
    assert(nk >= 1);
    if (nk < model.batchsize && ~batchsize_warning)
      fprintf('g=%g Going to batchsize <= %d due to memory limit.\n', gpu.FreeMemory/8, nk);
      batchsize_warning=1;
    end

    j = min(nx, i + nk - 1);
    ij = j-i+1;

    ti=0;model.ttimes=model.ttimes+1;
    ti=1;wait(gpuDevice); model.t(ti) = model.t(ti)+toc();

    val_f=zeros(nc, ij, 'gpuArray');
    if ns>0                   % 484802us
      xij=gpuArray(X(:,i:j));           % 27027us for batchsize=1250
      svi=1;
      for svblock=1:numel(SVtr)         % 219109us

      ti=0;model.ttimes2=model.ttimes2+1;
      ti=20;wait(gpuDevice);model.t(ti)=model.t(ti)+toc;
        sv = SVtr{svblock};             % 1727us
      ti=21;wait(gpuDevice);model.t(ti)=model.t(ti)+toc;
        svj = svi + size(sv, 1) - 1;    % 929us
      ti=22;wait(gpuDevice);model.t(ti)=model.t(ti)+toc;
        val_f = val_f + model.beta(:,svi:svj) * (hp.gamma * full(sv * xij) + hp.coef0) .^ hp.degree; % 166061us
      ti=23;wait(gpuDevice);model.t(ti)=model.t(ti)+toc;
        svi = svj + 1;
      end
      clear xij;
      assert(svj == ns);
      if model.b~=0 val_f = model.b + val_f; end
    end % if ns>0

    ti=2;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
      
    Yi = gpuArray(int32(Y(i:j)) + model.n_cla*int32(0:ij-1)); % 1018us
    tmp=val_f;                            % 922us
    tmp(Yi)=-inf;                         % 1200us
    [mx_val,idx_mx_val]=max(gather(tmp));         % 983us
    clear tmp;

    % TODO: figure out how to calculate these correctly in minibatch:
    % model.errTot=model.errTot+gather(sum(val_f(Yi)<=mx_val)); % 1186us
    % model.aer(model.iter)=model.errTot/(model.iter * model.batchsize);
    % model.pred(:,model.iter)=val_f;

    tr_val = val_f(Yi);                   % 996us
    clear val_f Yi;

    updates = find(tr_val <= mx_val);     % 1219us
    clear tr_val mx_val;

    ti=9;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    if ~isempty(updates)                % 33587us

      %ti=0;model.ttimes2 = model.ttimes2 + 1;
      %ti=20;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      updates_i = updates+i-1;              % 
      model.S = [model.S updates_i];        % 969us

      % Add new block to SVtr if necessary.
      % Keep it to two blocks.
      nu = numel(updates_i);
      assert(nu < sv_block_size);
      if nu + size(SVtr{end}, 1) > sv_block_size
        assert(numel(SVtr) <= 2);
        if (numel(SVtr) == 2)
          sv1 = [ gather(SVtr{1}); gather(SVtr{2}) ];
          clear SVtr{1}; clear SVtr{2}; wait(gpuDevice);
          SVtr{1} = gpuArray(sv1);
        end
        SVtr{2} = zeros(0, nd, 'gpuArray');
      end

      SVtr{end} = [ SVtr{end}; X(:,updates_i)' ];

      ns = ns + nu;

      newbeta = zeros(nc, nu, 'gpuArray'); % 982us
      mx_val_updates = gpuArray(int32(idx_mx_val(updates)) + nc*int32(0:nu-1)); % 1171us
      tr_val_updates = gpuArray(int32(Y(updates_i)) + nc*int32(0:nu-1)); % 1157us
      newbeta(tr_val_updates) = 1;          % 1144us
      newbeta(mx_val_updates) = -1;         % 1141us

      model.beta2 = model.beta2 + model.beta;
      model.beta2 = [model.beta2 newbeta];
      model.beta = [model.beta newbeta];    % 972us

      clear newbeta mx_val_updates tr_val_updates updates_i;

    end % if ~isempty(updates)

    clear updates;

    ti=10;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    if mod(model.iter,model.step)==0      % 1037us
      fprintf('#%.0f g:%g nk:%d SV:%5.2f(%d)\tAER:%5.2f\tt=%g\n', ...
              j, gpu.FreeMemory/8,nk,numel(model.S)/j*100,numel(model.S),model.aer(model.iter)*100,toc());
    end % if                                  % 881μs/916μs

    i = j+1;

  end % while i <= nx

end % for epoch=1:model.epochs

fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\tt=%g\n', ...
        j,numel(model.S)/j*100,numel(model.S),model.aer(model.iter)*100,toc());

model.beta = gather(model.beta);
model.beta2 = gather(model.beta2);
model.S = gather(model.S);
model.SV = [];
for svblock=1:numel(SVtr)
  model.SV = [ model.SV gather(SVtr{svblock})' ];
  clear SVtr{svblock};
end

end % k_perceptron_multi_train_gpu
