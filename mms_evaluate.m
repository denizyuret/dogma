function output = mms_evaluate(X, S, Y, model)
% MMS_EVALUATE  Max Margin Set learning algorithm auxiliary function
%
%    MODEL = MMS_EVALUATE(X, S, Y, MODEL)
%
%    Input: 
%
%    Additional parameters:
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


N = numel(X);
D = size(X{1}, 1);
K = model.n_cla;
W = reshape(model.W, D, K);

corr_pred = 0; corr_set = 0; total = 0; loss = 0;
for i=1:N
  Xt = X{i};
  St = S{i};

  Zt = unique(St);
  Kt = numel(Zt);
  Mt = size(Xt, 2);
  Lt = size(St, 1);

  val_f = full(W'*Xt);

  % calculate the objective function
  [ y_mx_pos margin_mx_pos ] = mx_pos_set(val_f, St);
  [ y_mx_vio loss1 ] = mx_vio_label_vec(val_f, St, margin_mx_pos);
  loss = loss + max(0, loss1);

  if ~isempty(Y)
    % prediction 
    [ dummy, y_pred ] = max(val_f);
    corr_pred = corr_pred + numel(find(y_pred == Y{i}));

    y_set     = mx_pos_set(val_f, St);
    corr_set  = corr_set + numel(find(y_set == Y{i}));

    total= total + Mt;
  end
end

output.loss = loss;
output.obj  = model.lambda*norm(model.W, 2)^2/2+loss/N;

if ~isempty(Y)
  output.acc_pred = corr_pred/total*100;
  output.acc_set  = corr_set/total*100;
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
  Zj    = unique(S(:, j));  
  Z(j)  = numel(Zj);
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

