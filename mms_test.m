function [ ypred yset accpred accset ] = mms_test(X, S, Y, model)
% MMS_TEST  Max Margin Set learning algorithm auxiliary function
%
%    MODEL = MMS_TEST(X, S, Y, MODEL)
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


K = model.n_cla;
if iscell(X)
   D = size(X{1}, 1);
else
   D = size(X, 1);
end

if size(model.W, 2)==1
   if model.bias
      W = reshape(model.W, D+1, K);
      b = W(end, :);
      W = W(1:end-1, :);
   else
      W = reshape(model.W, D, K);
   end
else
   W = model.W;
   if model.bias
      b = model.b;
   end
end

if iscell(X)
   N = numel(X);
   
   ypred = cell(N, 1);
   yset  = cell(N, 1);
   
   y = [];
   yhatp = [];
   yhats = [];
   
   for i=1:N   
     % prediction
     val_f = W'*X{i};
     if model.bias
        val_f = val_f + repmat(b', 1, size(X{i}, 2));
     end
     [  dummy, ypred{i} ] = max(val_f);
     [ yset{i} margins ]  = mx_pos_set(val_f, S{i});

     y = [ y Y{i} ];
     yhatp = [ yhatp ypred{i} ];
     yhats = [ yhats yset{i}  ];
   end
else
   N = size(X, 2);

   val_f = W'*X;
   if model.bias
      val_f = val_f + repmat(b', 1, N);
   end

   [ dummy, ypred ] = max(val_f);

   y     = Y;
   yhatp = ypred;
end

accpred(1) = numel(find(yhatp == y))/numel(y)*100;
classes = unique(y);
acc_cls = [];
for k=1:numel(classes)
   idx = find(y == classes(k));
   acc_cls(k) = numel(find(yhatp(idx) == y(idx)))/numel(idx)*100;
end
accpred(2) = mean(acc_cls);

if ~isempty(S) 
  accset(1) = numel(find(yhats == y))/numel(y)*100;
  acc_cls = [];
  for k=1:numel(classes)
     idx = find(y == classes(k));
     acc_cls(k) = numel(find(yhats(idx) == y(idx)))/numel(idx)*100;
  end
  accset(2) = mean(acc_cls);
else
  yset   = []; 
  accset = [];
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
