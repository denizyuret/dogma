function model = mms_multi_train(X, S, Y, model, Xtest, Stest, Ytest)
% MMS_MULTI_TRAIN  Max Margin Set learning algorithm
%
%    MODEL = MMS_MULTI_TRAIN(X, S, Y, MODEL) trains a classifier using the
%    Max Margin Set learning algorithm.
%
%    Input: 
%    X -  Training data: 1*N cell matrix, each cell X{i} is a D*Mi matrix,
%         each column correspond to a vector.
%    S -  Possible lable sets: 1*N cell matrix, each cell S{i} is a Li*Mi
%         matrix, each rows correpsond a set of possible labels Li is the
%         number of label sets.
%    Y -  True label: 1*N cell matrix, each cell Y{i} is a Mi dimension
%         vector, each element correspond to an instance in X{i}.
%
%    Additional parameters:
%    - model.k
%    - model.lambda
%    - model.R
%    - model.T
%    - model.bias
%
% Example:
%     See demos/demo_mms.m
%
% Reference:
%     - Jie, L. & Orabona, F. (2010)
%       Learning from Candidate Labeling Sets. 
%       In Advances in Neural Information Processing Systems 23 (NIPS10).

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
%    Contact the authors: jluo      [at] idiap.ch
%                         francesco [at] orabona.com


timerstart = cputime;

N = numel(X);
D = size(X{1}, 1);
K = model.n_cla;

sparseflag = issparse(X{1});

if isfield(model,'iter')==0
    model.iter     = 0;
    model.acc      = [];
    model.time     = [];
    model.round    = [];
    model.outputER = [];
end

if isfield(model,'k')==0
    model.k    = 1;
    model.proj = 0;
end

if isfield(model,'R')==0
    model.R = 5;
end

if isfield(model,'T')==0
    model.T = 100;
end

if isfield(model,'lambda')==0
    model.lambda = 1/N;
end

if isfield(model,'step')==0
    model.step = Inf;
end

if isfield(model, 'bias')==0
   model.bias = 0;
else
   % including bias term by increasing the dimension
   if model.bias
     for i=1:N
       X{i}(end+1, :)=1;
     end
     D = D+1;
   end
end

if isfield(model, 'proj')==0
   model.proj=0;
end

Mmax = 0;
for i=1:N
  Mi = size(X{i}, 2);
  if Mi>Mmax
    Mmax = Mi;
  end
end

if isfield(model, 'W')==0
   if sparseflag
     W = spalloc(K*D, 1, K*D);
   else
     W = zeros(K*D, 1);
   end
else
   W = model.W;
end

% CCCP parameter
for round=1:model.R
  fprintf('CCCP Round: %d\n', round);  

  % compute beta: 1st-order talyor coefficient of max_{Z \in  A} phi(X, Z) 
  % and the tranform the expansions 
  beta     = cell(N, 1);
  Wt       = reshape(W, D, K);

  %PhiXZpos = spalloc(K*D, N, Mi*D*N);
  PhiXZpos = spalloc(K*D, N, Mi*D*N);
  for i=1:N
    Xt = X{i};
    St = S{i};
    Mt = size(Xt, 2);

    val_f = Wt'*Xt;

    [ y_pos_set y_pos_idx ] = mx_pos_sets(val_f, St);

    betaij  = 1/numel(y_pos_idx);
    beta{i} = y_pos_idx; 
    Xt = betaij*Xt;

    % create max_{Z \in  A} phi(X, Z)
    for j=1:Mt
      Ztj = unique(y_pos_set(:,j));
      for k=1:numel(Ztj)
        f = numel(find(y_pos_set(:,j)==Ztj(k)));
        if f>0
          Ztj_k_D=Ztj(k)*D;
          PhiXZpos(Ztj_k_D-D+1:Ztj_k_D, i) = ...
          PhiXZpos(Ztj_k_D-D+1:Ztj_k_D, i) + f*Xt(:, j);
        end
      end
    end
  end

  % optimize the new optimization problem
  model.iter = 0;
  cumfactor = 1;
  for epoch=1:model.T
    idx_rand=randperm(N);

    for t=1:model.k:(N-model.k+1)
        model.iter=model.iter+1;

        idxs_for_subgrad=idx_rand(t:t+model.k-1);

        % pegasos update
        eta    = 1/(model.lambda*model.iter);
        factor = 1-model.lambda*eta;
        % only multiple W with the factor when an update is performed, 
        % otherwise cache it
        cumfactor = cumfactor*factor;
        
        update = false;
        for i=1:model.k
           Xt = X{idxs_for_subgrad(i)};
           St = S{idxs_for_subgrad(i)}; 

           Mt = size(Xt, 2);

           val_f = full(Wt'*Xt);
           [ y_mx_pos, margin_pos ] = mx_pos_set(val_f, St);

           % create the mapped vector max_{Z \notin  A} phi(X, Z)
           [ y_mx_vio loss ] = mx_vio_label_vec(val_f, St, margin_pos);	   
            
           % update
           if loss>0
              update = true;              
              if cumfactor ~= 1
                W = W*cumfactor; 
                cumfactor = 1;
              end                         
              W = W + eta/model.k * PhiXZpos(:, idxs_for_subgrad(i));
              for j=1:Mt
                y_mx_vio_j_D=y_mx_vio(j)*D;
                W(y_mx_vio_j_D-D+1:y_mx_vio_j_D) = ...
                W(y_mx_vio_j_D-D+1:y_mx_vio_j_D) - eta/model.k * Xt(:, j);
              end              
           end
        end
        
        if update 
          if model.proj
             W = min(1, (sqrt(2*Mmax/model.lambda))/norm(W, 2)) * W;  
          end
          Wt = reshape(W, D, K);
        end        
    end
        
    model.time(end+1) = cputime-timerstart;
    if mod(epoch, model.step)==0 || epoch == model.T
      model.W = W;
      output = mms_evaluate(X, S, Y, model);
      if isempty(model.outputER)
         model.outputER = output;
      else
         model.outputER(end+1) = output; 
      end
      if exist('Xtest') && ~isempty(Xtest)
         [ ypred yset accpred accset ] = mms_test(Xtest, Stest, Ytest, model); 
         fprintf('\tEpoch %d:\t AccPred=%.2f  AccSet=%.2f  Obj=%.2f  Loss=%.2f  AccTestPred=%.2f/%.2f', ...
                  epoch, output.acc_pred, output.acc_set, output.obj, output.loss, accpred(1), accpred(2));   
         if ~isempty(Stest)
            fprintf('\tAccTestSet=%.2f/%.2f\n', accset(1), accset(2));
         else
            fprintf('\n');
         end 
	     model.acc(:, end+1) = [accpred accset];
      end
    end
    timerstart = cputime;
  end %end of pegasos
  
  if cumfactor ~= 1
    W = W*cumfactor; 
  end      
end  %end of CCCP

W = reshape(W, D, K);
if model.bias
  model.W = W(1:end-1, :);
  model.b = W(end, :);
else
  model.W = W;
end

% =============== built in functions ===============
% mx_pos_set
function [ y margin ] = mx_pos_set(val_f, S)
% find the (one) set with maximal sum margins in S
L = size(S, 1);
M = size(S, 2);

[ v_hat y_hat ] = max(val_f);
if ismember(y_hat, S, 'rows')
  y      = y_hat;
  margin = sum(v_hat);
  return
else
  margin_set = zeros(L, 1);
  for l=1:L
    for j=1:M
      margin_set(l) = margin_set(l)+val_f(S(l,j) ,j);
    end
  end
  [ dummy, idx ] = sort(margin_set, 'descend');

  y      = S(idx(1), :);
  margin = margin_set(idx(1));
end

% ------------------------------------------------
% mx_pos_sets
function  [ y_set y_idx ] = mx_pos_sets(val_f, S)
% find the sets with maximal sum margins in S
L = size(S, 1);
M = size(S, 2);

margin_set = zeros(L, 1);
for l=1:L
  for j=1:M
    margin_set(l) = margin_set(l)+val_f(S(l,j) ,j);
  end
end

[ margin_set, idx ] = sort(margin_set, 'descend');
S = S(idx, :);

y_idx = [];
y_set = [];
for l=1:L
  y_set = [ y_set; S(l, :) ];
  y_idx(end+1, 1) = idx(l);
  if l==L || margin_set(l)>margin_set(l+1)
    return 
  end
end

% ------------------------------------------------
% mx_vio_label_vec
function [ y maxloss hl ] = mx_vio_label_vec(val_f, S, m_pos)
% find the sets with maximal loss (structure hamming loss) not in S
% worst case complexity max(O(L*M^2),  O(ZlogZ))
K = size(val_f, 1);
L = size(S, 1);
M = size(S, 2);

% add hamming loss to the margin
Z = zeros(1, M);
Carray = 1:K;
for j=1:M
  Zj   = unique(S(:, j));  
  Z(j) = numel(Zj);
  %idx = setdiff(1:K, Zj);
  idx  = Carray(~ismembc(Carray, Zj));
  % margin rescaling, hamming loss
  val_f(idx, j) = val_f(idx, j)+1;
end

[ m_sort y_sort ] = sort(val_f, 'descend');
y_sets = y_sort(1:max(Z)+1, :);
m_sets = m_sort(1:max(Z)+1, :);

p_stack = zeros((L+1)*M, M);
v_stack = zeros((L+1)*M, 1);
% indexes of used stack
idxend = 1;
i_stack = idxend;
% index which point to the first column of the sorted set
p_stack(1, :) = ones(1, M) + cumsum([0 (max(Z)+1)*ones(1, M-1)]);
v_stack(1, 1) = sum(m_sets(1, :));

iter = 0;
while iter <= L+1
   [ m_mx m_idx ] = max(v_stack(i_stack));
   idx  = p_stack(i_stack(m_idx), :);
   y_mx = y_sets(idx);   
   if ~ismember(y_mx, S, 'rows')
     y       = y_mx;
     maxloss = m_mx - m_pos;
     return
   else
     i_stack(m_idx) = [];
     % continue on the path
     for j=1:M
       idxend = idxend+1;
       i_stack(end+1) = idxend;
       newidx     = idx;
       newidx(j)  = newidx(j)+1;
       v_stack(idxend, 1) = sum(m_sets(newidx));
       p_stack(idxend, :) = newidx;
     end
   end
   iter = iter+1;
end
error('Did not find the most violate labeling vector');

