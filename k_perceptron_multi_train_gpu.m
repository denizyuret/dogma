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

n = length(Y);   % number of training samples

if isfield(model,'n_cla')==0
  model.n_cla=max(Y);
end

if isfield(model,'iter')==0
  model.iter=0;
  model.beta=zeros(model.n_cla, 0, 'gpuArray');
  model.beta2=zeros(model.n_cla, 0, 'gpuArray');
  model.S = zeros(1, 0, 'gpuArray');
  model.SV = zeros(size(X, 1), 0, 'gpuArray');
  model.errTot=0;
  model.numSV=zeros(numel(Y),1);
  model.aer=zeros(numel(Y),1);
  model.pred=zeros(model.n_cla,numel(Y));
else
  assert(isfield(model,'ker'), 'Cannot continue training using a Kernel matrix as input.');
end

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
  model.batchsize=1;
end

if isfield(model,'epochs')==0
  model.epochs = 1;
end

tic();
for epoch=1:model.epochs
  % we should shuffle here?

  for i=1:model.batchsize:n               % 26986us/iter to process X(:,i:i+500)
    j = min(n, i + model.batchsize - 1);
    ij = j-i+1;
    model.iter=model.iter+1;
    %if model.iter == 3 return; end
    %fprintf('iter=%d i=%d j=%d\n', model.iter, i, j);

    model.ttimes=model.ttimes+1;
    %ti=1;wait(gpuDevice); model.t(ti) = model.t(ti)+toc();

    if numel(model.S)>0                   % 8065us, 3289479us

      model.ttimes2 = model.ttimes2 + 1;
      %ti=20;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
      if isempty(model.ker)
        K_x=X(model.S,i:j);
      else
        K_x=feval(model.ker,model.SV,X(:,i:j),model.kerparam); % 5639us (for batchsize=500,dim=804,nsv<2000)
                                                               % 2896019us (for batch=1250,dim=804,nsv~100K)
      end

      %ti=21;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
      val_f=model.beta*K_x;               % 985us (500x2000), 354539us (1250x100K)
      %ti=22;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
      clear K_x;

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
    Yi = gpuArray(int32(Y(i:j)) + model.n_cla*int32(0:ij-1)); % 1018us
                                                              %fprintf('Yi=%s/%s (%d,%d)\n', class(Yi), classUnderlying(Yi), size(Yi));
                                                              %dbg_yij = Y(i:j)
                                                              %dbg_yi = Yi
    %ti=3;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    tmp=val_f;                            % 922us
    %ti=4;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    tmp(Yi)=-inf;                         % 1200us
                                          %fprintf('tmp=%s/%s (%d,%d)\n', class(tmp), classUnderlying(tmp), size(tmp));
                                          %dbg_tmp = tmp
    %ti=5;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    [mx_val,idx_mx_val]=max(gather(tmp));         % 983us
    clear tmp;
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
    
    tr_val = val_f(Yi);                   % 996us
    %ti=8;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    clear val_f Yi;
    %ti=9;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    
    updates = find(tr_val <= mx_val);     % 1219us
    %ti=10;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    if ~isempty(updates)                % 299614usbig

      model.ttimes3 = model.ttimes3 + 1;
      %ti=30;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
      updates_i = updates+i-1;              % 
      %ti=31;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      model.S = [model.S updates_i];        % 969us
                                            %dbg_S = model.S
      %ti=32;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

      if ~isempty(model.ker)
        model.SV = [model.SV X(:,updates_i)]; % 1921us, 281407usbig
      end
      %fprintf('model.SV=%s/%s (%d,%d)\n', class(model.SV), classUnderlying(model.SV), size(model.SV));
      %ti=33;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

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

    %   % if model.ttimes > 2 return; end
    %   if 0

    %   for k=1:numel(tr_val)                 % 2921103us
    %     model.ttimes3 = model.ttimes3 + 1;
    %     ti=30;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %     jj = i + k - 1;
    %     % if val_f(Yi)<=mx_val
    %     if tr_val(k) <= mx_val(k)           % 3072us
    %       model.ttimes4 = model.ttimes4 + 1;
    %       ti=40;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %       model.S(end+1)=jj;          % 1058us
    
    %       %fprintf('model.S=%s/%s (%d,%d)\n', class(model.S), classUnderlying(model.S), size(model.S));
    %       %dbg_S = model.S

    %       ti=41;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %       if ~isempty(model.ker)
    %         model.SV(:,end+1)=X(:,jj);         % 1440us
    %       end
    %       %fprintf('model.SV=%s/%s (%d,%d)\n', class(model.SV), classUnderlying(model.SV), size(model.SV));
    %       %fprintf('model.SV(:,%d)=X(:,%d)\n', size(model.SV,2), jj);
    %       ti=42;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    %       model.beta(:,end+1)=zeros(model.n_cla,1);% 1044us
    %       %fprintf('model.beta=%s/%s (%d,%d)\n', class(model.beta), classUnderlying(model.beta), size(model.beta));
    %       %dbg_beta = model.beta
    %       ti=43;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    %       if model.update==1
    %         % max-score
    %         model.beta(Y(jj),end)=1;             % 1026us
    %         ti=44;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %         model.beta(idx_mx_val(k),end)=-1;    % 1019us
    %         ti=45;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %       else
    %         % uniform
    %         model.beta(:,end)=-1/(model.n_cla-1);
    %         model.beta(Y(jj),end)=1;
    %       end
    %       ti=46;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %       %fprintf('model.beta=%s/%s (%d,%d)\n', class(model.beta), classUnderlying(model.beta), size(model.beta));
    %       %dbg_beta_updated = model.beta
    
    %       if model.maxSV==inf
    %         model.beta2(:,end+1)=zeros(model.n_cla,1); % 1049us
    %       end
    %       ti=47;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %       %fprintf('model.beta2=%s/%s (%d,%d)\n', class(model.beta2), classUnderlying(model.beta2), size(model.beta2));
    %       %dbg_beta2 = model.beta2
    
    %       if numel(model.S)>model.maxSV
    %         mn_idx=ceil(model.maxSV*rand);
    %         model.beta(:,mn_idx)=[];
    %         if isfield(model,'ker')
    %           model.SV(:,mn_idx)=[];
    %         end
    %         model.S(mn_idx)=[];
    %       end %if 
    %       ti=48;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %     end %   if val_f(Yi)<=mx_val
    %     ti=31;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %     if model.maxSV==inf
    %       model.beta2=model.beta2+model.beta; % 973us
    %     end
    %     ti=32;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %     %fprintf('model.beta2=%s/%s (%d,%d)\n', class(model.beta2), classUnderlying(model.beta2), size(model.beta2));
    %     %dbg_beta2_updated = model.beta2
    
    %     model.numSV(jj)=numel(model.S); % 908us
    %     ti=33;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    
    %   end % for k=1:numel(tr_val)

    %   ti=9;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 
    %   assert(jj == j);                      % 927us

    %   end % if 0

    %ti=11;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

    if mod(model.iter,model.step)==0      % 1037us
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\tt=%g\n', ...
              j,numel(model.S)/j*100,numel(model.S),model.aer(model.iter)*100,toc());
    end % if                                  % 881μs/916μs

    %ti=12;wait(gpuDevice); model.t(ti) = model.t(ti)+toc(); 

  end % for i=1:model.batchsize:n
end % for epoch=1:model.epochs

fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\tt=%g\n', ...
        j,numel(model.S)/j*100,numel(model.S),model.aer(model.iter)*100,toc());

model.beta = gather(model.beta);
model.beta2 = gather(model.beta2);
model.S = gather(model.S);
model.SV = gather(model.SV);

end % k_perceptron_multi_train_gpu
