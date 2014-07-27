function model = k_perceptron_multi_train_gpu(X,Y,model)
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

% Make sure we have a poly kernel model
assert(isfield(model,'ker') && ~isempty(model.ker) && isfield(model,'kerparam'),...
       'Only kernel models supported.\n');
hp = model.kerparam;
assert(strcmp(hp.type,'poly'), 'Only poly kernel models supported.\n');

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
sv_buffer_size = 10000;

if isfield(model,'iter')==0
  fprintf('Initializing new model.\n');
  model.iter=0;
  model.beta=zeros(nc, 0, 'gpuArray');
  model.beta2=zeros(nc, 0, 'gpuArray');
  model.S = zeros(1, 0, 'gpuArray');
  SVtr1 = [];
  SVtr2 = zeros(sv_buffer_size, nd, 'gpuArray');
  model.errTot=0;
  model.numSV=zeros(nx,1);
  model.aer=zeros(nx,1);
  model.pred=zeros(nc,nx);
  ns1 = 0;
  ns2 = 0;
  ns = ns1 + ns2;
else
  assert(isfield(model,'ker'), ['Cannot continue training using a Kernel matrix as input.']);
  fprintf('g=%g Loading model to gpu...\n', gpu.FreeMemory/8);tic;
  model.beta = gpuArray(model.beta);
  model.beta2 = gpuArray(model.beta2);
  model.S = gpuArray(model.S);
  SVtr1 = gpuArray(model.SV');
  SVtr2 = zeros(sv_buffer_size, nd, 'gpuArray'); % stupid matlab copies on subasgn, need to keep separate
  ns1 = size(model.SV, 2);
  ns2 = 0;
  ns = ns1 + ns2;
  wait(gpuDevice);
  toc;
end
fprintf('g=%g Model on gpu...\n', gpu.FreeMemory/8);
toc;

if isfield(model,'update')==0
  model.update=1; % max-score
end

if isfield(model,'maxSV')==0
  model.maxSV=inf;
end

%if isfield(model,'t')==0
  model.t=zeros(1,100);
  model.ttimes=0;
  model.ttimes2=0;
  model.ttimes3=0;
  model.ttimes4=0;
%end

if isfield(model,'batchsize')==0
  model.batchsize=1000;
end
batchsize_warning = 0;
fprintf('Using batchsize=%d\n', model.batchsize);

if isfield(model,'epochs')==0
  model.epochs = 1;
end

tic();
for epoch=1:model.epochs
  % we should shuffle here?  it would make S indices meaningless.

  i = 1;
  while (i <= nx)               % 26986us/iter to process X(:,i:i+500)

    % compute the real batchsize here based on memory
    ns = size(model.beta,2);
    assert(ns == ns1+ns2);
    assert(ns1 == size(SVtr1, 1));
    assert(ns2 <= size(SVtr2, 1));
    nk = floor(0.9 * (gpu.FreeMemory/8) / (2*ns+2*nd+5*nc+10));
    nk = min(nk, model.batchsize);
    assert(nk >= 1);
    if (nk < model.batchsize && ~batchsize_warning)
      fprintf('g=%g Going to batchsize <= %d due to memory limit.\n', gpu.FreeMemory/8, nk);
      batchsize_warning=1;
    end

    j = min(nx, i + nk - 1);
    ij = j-i+1;
    model.iter=model.iter+1;

    %ti=0;model.ttimes=model.ttimes+1;
    %ti=1;wait(gpuDevice); model.t(ti) = model.t(ti)+toc();

    if numel(model.S)>0                   % 8065us, 3289479us

      
      %ti=0;model.ttimes2 = model.ttimes2 + 1;
      %ti=20;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      % K_x=feval(model.ker,model.SV,X(:,i:j),model.kerparam); % 5639us (for batchsize=500,dim=804,nsv<2000)
                                                               % 2896019us (for batch=1250,dim=804,nsv~100K)

      % wait(gpuDevice);fprintf('g=%g calculating val_f.\n', gpu.FreeMemory/8);

      if ns1 > 0
        k1 = (hp.gamma * full(SVtr1 * X(:,i:j)) + hp.coef0) .^ hp.degree;
      else 
        k1 = [];
      end
      if ns2 > 0
        k2 = (hp.gamma * full(SVtr2(1:ns2,:) * X(:,i:j)) + hp.coef0) .^ hp.degree;
      else
        k2 = [];
      end
      val_f = model.b + model.beta * [k1; k2];
      clear k1 k2;

      %ti=21;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
      %val_f=model.beta*K_x;               % 985us (500x2000), 354539us (1250x100K)
      %ti=22;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
      %     val_f = zeros(model.n_cla, model.batchsize, 'gpuArray');
      %     %fprintf('val_f=%s/%s (%d,%d)\n', class(val_f), classUnderlying(val_f), size(val_f));
      %     %fprintf('K_x=%s/%s (%d,%d)\n', class(K_x), classUnderlying(K_x), size(K_x));
      %     %fprintf('model.beta=%s/%s (%d,%d)\n', class(model.beta), classUnderlying(model.beta), size(model.beta));
      %     for c=1:model.n_cla
      %       val_f(c,:)=dot(model.beta(c,:),K_x);
      %     end                                 % 2855μs/2432μs
      %fprintf('K_x=%s/%s (%d,%d)\n', class(K_x), classUnderlying(K_x), size(K_x));

    else
      val_f=zeros(model.n_cla, ij, 'gpuArray');
    end % if numel(model.S)>0
    
    %fprintf('val_f=%s/%s (%d,%d)\n', class(val_f), classUnderlying(val_f), size(val_f));
    %dbg_vf = val_f
    %ti=2;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    % Yi=Y(i);

    % wait(gpuDevice);fprintf('g=%g calculating Yi.\n', gpu.FreeMemory/8);
      
    Yi = gpuArray(int32(Y(i:j)) + model.n_cla*int32(0:ij-1)); % 1018us

    %fprintf('Yi=%s/%s (%d,%d)\n', class(Yi), classUnderlying(Yi), size(Yi));
    %dbg_yij = Y(i:j)
    %dbg_yi = Yi
    %ti=3;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    % wait(gpuDevice);fprintf('g=%g calculating tmp.\n', gpu.FreeMemory/8);

    tmp=val_f;                            % 922us

    %ti=4;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    tmp(Yi)=-inf;                         % 1200us

    %fprintf('tmp=%s/%s (%d,%d)\n', class(tmp), classUnderlying(tmp), size(tmp));
    %dbg_tmp = tmp
    %ti=5;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    % wait(gpuDevice);fprintf('g=%g getting max tmp.\n', gpu.FreeMemory/8);

    [mx_val,idx_mx_val]=max(gather(tmp));         % 983us
    clear tmp;

    % wait(gpuDevice);fprintf('g=%g cleared tmp.\n', gpu.FreeMemory/8);

    %ti=6;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %fprintf('mx_val=%s (%d,%d)\n', class(mx_val), size(mx_val));
    %dbg_mx_val = mx_val
    %fprintf('idx_mx_val=%s (%d,%d)\n', class(idx_mx_val), size(idx_mx_val));
    %dbg_idx_mx_val = idx_mx_val

    model.errTot=model.errTot+gather(sum(val_f(Yi)<=mx_val)); % 1186us

    %dbg_errTot = model.errTot

    model.aer(model.iter)=model.errTot/(model.iter * model.batchsize);

    %dbg_aer = model.aer(model.iter)
    % model.pred(:,model.iter)=val_f;
    %ti=7;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    
    % wait(gpuDevice);fprintf('g=%g calculating tr_val.\n', gpu.FreeMemory/8);

    tr_val = val_f(Yi);                   % 996us

    %ti=8;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    % wait(gpuDevice);fprintf('g=%g clear val_f Yi.\n', gpu.FreeMemory/8);

    clear val_f Yi;

    %ti=9;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    
    % wait(gpuDevice);fprintf('g=%g updates.\n', gpu.FreeMemory/8);

    updates = find(tr_val <= mx_val);     % 1219us

    % wait(gpuDevice);fprintf('g=%g clear tr_val.\n', gpu.FreeMemory/8);

    clear tr_val;

    %ti=10;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    if ~isempty(updates)                % 299614usbig

      %ti=0;model.ttimes3 = model.ttimes3 + 1;
      %ti=30;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      % wait(gpuDevice);fprintf('g=%g updates_i.\n', gpu.FreeMemory/8);

      updates_i = updates+i-1;              % 

      %ti=31;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      % wait(gpuDevice);fprintf('g=%g model_S.\n', gpu.FreeMemory/8);

      model.S = [model.S updates_i];        % 969us
                                            %dbg_S = model.S
      %ti=32;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      % wait(gpuDevice);fprintf('g=%g checking size model_SVtr.\n', gpu.FreeMemory/8);

      % Grow SVtr2 if necessary
      nu = numel(updates_i);
      if ns2 + nu > size(SVtr2, 1)
        SVtr2 = [SVtr2; zeros(nu + sv_buffer_size, nd)];
        wait(gpuDevice);
      end

      % wait(gpuDevice);fprintf('g=%g udpating model_SVtr.\n', gpu.FreeMemory/8);
      SVtr2(ns2+1:ns2+nu,:) = X(:,updates_i)'; % 1921us, 281407usbig
      ns2 = ns2 + nu;
      ns = ns + nu;

      %fprintf('model.SV=%s/%s (%d,%d)\n', class(model.SV), classUnderlying(model.SV), size(model.SV));
      %ti=33;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      % wait(gpuDevice);fprintf('g=%g newbeta.\n', gpu.FreeMemory/8);

      newbeta = zeros(model.n_cla, numel(updates), 'gpuArray'); % 982us

      %ti=34;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      mx_val_updates = gpuArray(int32(idx_mx_val(updates)) + model.n_cla*int32(0:numel(updates)-1)); % 1171us

      %ti=35;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      tr_val_updates = gpuArray(int32(Y(updates_i)) + model.n_cla*int32(0:numel(updates)-1)); % 1157us

      %ti=36;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      newbeta(tr_val_updates) = 1;          % 1144us

      %ti=37;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      newbeta(mx_val_updates) = -1;         % 1141us

      %ti=38;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
      %fprintf('newbeta=%s/%s (%d,%d)\n', class(newbeta), classUnderlying(newbeta), size(newbeta));
      %fprintf('model.beta=%s/%s (%d,%d)\n', class(model.beta), classUnderlying(model.beta), size(model.beta));

      model.beta2 = model.beta2 + model.beta;

      %ti=39;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      model.beta2 = [model.beta2 newbeta];

      %ti=40;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      model.beta = [model.beta newbeta];    % 972us
                                            %dbg_model_beta = model.beta
      %ti=41;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      clear newbeta mx_val_updates tr_val_updates updates_i;

      %ti=42;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    end % if ~isempty(updates)

    clear updates;

    %ti=11;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    if mod(model.iter,model.step)==0      % 1037us
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\tt=%g\n', ...
              j,numel(model.S)/j*100,numel(model.S),model.aer(model.iter)*100,toc());
    end % if                                  % 881μs/916μs

    %ti=12;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    i = j+1;

  end % while i <= nx

end % for epoch=1:model.epochs

fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\tt=%g\n', ...
        j,numel(model.S)/j*100,numel(model.S),model.aer(model.iter)*100,toc());

model.beta = gather(model.beta);
model.beta2 = gather(model.beta2);
model.S = gather(model.S);
model.SV = [gather(SVtr1)' gather(SVtr2(1:ns2,:))'];
clear SVtr1 SVtr2;

end % k_perceptron_multi_train_gpu
