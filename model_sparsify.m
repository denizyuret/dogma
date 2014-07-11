function m = model_sparsify(X,Y,model,average)
% MODEL_SPARSIFY implements an algorithm that reduces the number of
% support vectors in a kernel based model.  Handles both binary and
% multiclass models.  Uses the averaged solution (beta2) if available
% by default.  Does not handle pre-computed kernel matrices.
%
%    MODEL = MODEL_SPARSIFY(X,Y,MODEL0,AVERAGE) takes a model already 
%    trained on X,Y and returns one with fewer support vectors.
%
%    Additional parameters: 
%
%    - model.eta is the learning rate.  Default is 0.5.  It will be
%    multiplied by mean(beta(:)) for scaling.
%
%    - model.epsilon is the margin requirement.  Default is 0.5.  It
%    will be multiplied by mean(margins(:)) for scaling.  The
%    algorithm will stop when the approximate solution achieves a
%    margin for all training vectors (solved correctly by the
%    original model) that is at least min(epsilon, m-epsilon) where
%    m is the margin of the original solution.
%
%   References:
%     - Cotter, A.; Shalev-Shwartz, S.; Srebro, N. (2013).  Learning
%     Optimally Sparse Support Vector Machines. ICML.
%
%    This file is part of the DOGMA library for MATLAB.
%    Copyright (C) 2009-2011, Francesco Orabona; 2014 Deniz Yuret.
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
%    Contact the author: denizyuret [at] gmail.com

% Make sure we have a kernel based model
assert(isfield(model,'beta'), 'Need kernel model');
assert(isfield(model,'ker') && ~isempty(model.ker), 'Need kernel model');

% Use averaged hyperplain (beta2) by default
if (nargin<4) average=1; end
if (~isfield(model,'beta2')) average=0; end
if average fprintf('Using averaged solution beta2.\n'); end

% Set default margin threshold and learning rate
if (isfield(model, 'epsilon')==0) model.epsilon = 0.5; end
if (isfield(model, 'eta')==0) model.eta = 0.5; end

% X: dxn training instances
% Y: 1xn training labels (1:c if multi, -1,+1 if binary)
% model.beta: cxs model coefficients
n = numel(Y);                   % n: number of training instances
c = size(model.beta, 1);	% c: number of classes if multi, 1 if binary
max_num_el=500*1024^2/8; % 500 Mega of memory as maximum size for K
xstep=ceil(max_num_el/size(model.beta, 2));
fprintf('Instances: %d, classes: %d, nsv: %d\n', n, c, size(model.beta,2));

% scores calculates the cxn score matrix for each class and each instance
function f=scores(x)
f = zeros(c, n);
for i=1:xstep:n
  K = feval(model.ker, model.SV, x(:,i:min(i+xstep-1,n)),model.kerparam);
  if average==0
    f(:,i:min(i+xstep-1,n)) = model.beta*K+model.b;
  else
    f(:,i:min(i+xstep-1,n)) = model.beta2*K+model.b2;
  end
  fprintf('.');
end
fprintf('\n');
end % scores

% predicted_labels takes the cxn score matrix and predicts labels
function p=predicted_labels(f)
  if c>1                        % multiclass
    [~, p] = max(f, [], 1);
  else                          % binary
    p = sign(f);
  end
end

% margins() takes f, a cxn matrix of scores
% returns m, a 1xn matrix of margins: f(yi,i) - max[y~=yi] f(y,i)
% returns z, a 1xn matrix of best wrong answers: argmax[y~=yi] f(y,i)
% for binary models m = y .* f and there is no z
% Yi: 1D indices of correct answers for multi
if (c>1) Yi = Y + c*[0:n-1]; end

function [d,z]=margins(f)
if (c == 1)
  d = Y .* f;
else
  fYi = f(Yi);                  % save all correct answers
  f(Yi) = -inf;                 % replace them with -inf
  [maxf,z] = max(f);            % find best wrong answers
  d = fYi - maxf;               % margin is the difference
end % if
end % margins

tic(); 
fprintf('Computing scores for the input model in %d chunks\n', ceil(n/xstep));
initial_scores = scores(X);
initial_error = numel(find(predicted_labels(initial_scores) ~= Y));
fprintf('Initial nsv=%d error=%d/%d\n', size(model.beta, 2), initial_error, n);
toc();

% computing target margins.  paper assumes svm with
% margin=1 and uses epsilon=0.5.  since our margins are on an
% arbitrary scale we will scale them with mean of positive
% margins.
target_margin = margins(initial_scores);
mean_margin = mean(target_margin(target_margin>0));
target_margin(target_margin<0) = -inf; % to ignore mistakes in target
epsilon_scaled = model.epsilon * mean_margin;
fprintf('Scaled epsilon = %f\n', epsilon_scaled);
max_target = epsilon_scaled * 2;  % paper uses eps=0.5 and caps targets at 1
target_margin(target_margin>max_target) = max_target;

% the paper suggests a learning rate of 0.5 for margin=1 and
% k(x,x)=1.  we will use mean_abs_beta to scale our learning rate.
if average == 0
  mean_abs_beta = mean(abs(model.beta(:)));
else
  mean_abs_beta = mean(abs(model.beta2(:)));
end
eta_scaled = model.eta * mean_abs_beta;
fprintf('Scaled eta = %f\n', eta_scaled);

% Prepare new model
m = model;
m.iter = 0;  % training iterations
m.S=[];      % 1xs support vector indices
m.SV=[];     % dxs support vectors
m.beta=[];   % cxs support vector weights
m.beta2=[];  % cxs averaged weights (unused)
m.pred=zeros(c,n); % cxn score for each training instance
m.numSV=[];  % tx1 number of SV for each iteration
m.aer=[];    % tx1 number of errors for each iteration
m.errTot = 0; % final number of errors
fprintf('iter\tnsv\terr\tmax(h-c)\n');

while 1
  m.iter=m.iter+1;
  [d,z] = margins(m.pred);
  mdiff = target_margin - d;

  [maxdiff, si] = max(mdiff(m.S)); % Aggressive version:
  if maxdiff > epsilon_scaled   % first check existing sv with violation
    xi = m.S(si);
  else                          % otherwise check all instances
    [maxdiff, xi] = max(mdiff);
    if maxdiff < epsilon_scaled
      break                     % quit if no violation
    end
    m.S(end+1) = xi;        % add new sv if violation
    m.SV(:,end+1) = X(:,xi);
    si = numel(m.S);
  end

  if c == 1                     % binary update
    d_beta = Y(xi) * eta_scaled;
  else                          % multiclass update
    d_beta = zeros(c, 1);
    d_beta(Y(xi)) = eta_scaled;
    d_beta(z(xi)) = -eta_scaled;
  end
  if si > size(m.beta, 2)
    assert(si == 1 + size(m.beta, 2));
    m.beta(:,si) = d_beta;
  else
    m.beta(:,si) = m.beta(:,si) + d_beta;
  end
  m.pred = m.pred + d_beta * feval(m.ker,m.SV(:,si),X,m.kerparam);

  m.errTot = numel(find(predicted_labels(m.pred) ~= Y));
  m.aer(m.iter) = m.errTot;
  m.numSV(m.iter)=numel(m.S);

  if mod(m.numSV(m.iter), m.step) == 0 && m.numSV(m.iter) ~= m.numSV(m.iter-1)
    fprintf('%d\t%d\t%d\t%g\n', m.iter, numel(m.S), m.errTot, maxdiff);
  end % if

end % while

fprintf('%d\t%d\t%d\t%g\n', m.iter, numel(m.S), m.errTot, maxdiff);

end % model_sparsify
