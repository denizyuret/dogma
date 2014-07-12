function m = model_sparsify(model,x_tr,y_tr,p)
% MODEL_SPARSIFY implements an algorithm that reduces the number of
% support vectors in a kernel based model.  Handles both binary and
% multiclass models.  Uses the averaged solution (beta2) if available
% by default.  Does not handle pre-computed kernel matrices.
%
%    m = model_sparsify(model,x_tr,y_tr,p) takes a model, a set of
%    instances to choose new support vectors from (x_tr,y_tr), and
%    some optional parameters (p) and returns m, a sparser the
%    model, reporting its train/test set accuracy.  It does not
%    modify the input model.  It works for any kernel based
%    model, binary or multi-class.
%
%    Define margin of a model for an instance as the difference
%    between the score of the correct answer and the score of the
%    closest alternative.  The algorithm tries to achive one of the
%    following conditions for each instance:
%
%    - If margin(model,i) <= 0, ignore instance i.
% 
%    - If 0 < margin(model,i) < p.margin, allow m to make a
%    margin violation at most p.epsilon worse than the original,
%    i.e. margin(m,i) >= margin(model,i) - p.epsilon
%
%    - If margin(model,i) > p.margin, we would like m to satisfy
%    margin(m,i) >= p.margin - p.epsilon.
%
%    Both p.margin and p.epsilon will be multiplied by the mean
%    of the original positive margins for scaling.
%
%    Additional parameters: 
%
%    - p.average determines if the averaged model (model.beta2) will
%    be used rather than the last model (model.beta).  Default is
%    1.
%
%    - p.x_te, p.y_te is an optional development set.  If
%    supplied the error on the development set will be reported.
%
%    - p.eta is the learning rate.  Default is 0.5.  It will be
%    multiplied by mean(abs(beta(:))) for scaling and determines
%    the increments applied to m.beta.
%
%    - p.epsilon determines the difference acceptable between the
%    original margins and the new margins.  Default is 0.5.
%   
%    - p.margin is the cap for the original margins.  Default is
%    1.0.
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
if (nargin < 4) p = []; end
if (~isfield(p,'average')) p.average = 1; end
if (~isfield(model,'beta2')) p.average=0; end
if p.average fprintf('Using averaged solution beta2.\n');
else fprintf('Using the last solution beta.\n'); end

% Set default margin threshold and learning rate
if (isfield(p, 'epsilon')==0) p.epsilon = 0.5; end
if (isfield(p, 'margin')==0) p.margin = 1.0; end
if (isfield(p, 'eta')==0) p.eta = 0.5; end
fprintf('epsilon=%g margin=%g eta=%g\n', p.epsilon, p.margin, p.eta);

% x_tr: dxn training instances
% y_tr: 1xn training labels (1:c if multi, -1,+1 if binary)
% model.beta: cxs model coefficients
n = numel(y_tr);                % n: number of training instances
c = size(model.beta, 1);	% c: number of classes if multi, 1 if binary
nsv = size(model.beta, 2);      % nsv: number of sv in original model
if (isfield(p, 'y_te')==0) n_te = 0; else n_te = numel(p.y_te); end
fprintf('train: %d, test: %d, classes: %d, nsv: %d\n', n, n_te, c, nsv);

% scores calculates the cxn score matrix for each class and each instance
max_num_el=500*1024^2/8; % 500 Mega of memory as maximum size for K

function f=scores(x,mo,avg)
nc = size(mo.beta, 1);
nx = size(x, 2);
nsv = size(mo.beta, 2);
f = zeros(nc, nx);
xstep=ceil(max_num_el/nsv);
for i=1:xstep:nx
  K = feval(mo.ker, mo.SV, x(:,i:min(i+xstep-1,nx)),mo.kerparam);
  if avg==0
    f(:,i:min(i+xstep-1,nx)) = mo.beta*K+mo.b;
  else
    f(:,i:min(i+xstep-1,nx)) = mo.beta2*K+mo.b2;
  end
end
end % scores

% predicted_labels takes the cxn score matrix and predicts labels
function l=predicted_labels(f)
  if c>1                        % multiclass
    [~, l] = max(f, [], 1);
  else                          % binary
    l = sign(f);
  end
end

% margins() takes f, a cxn matrix of scores
% returns m, a 1xn matrix of margins: f(yi,i) - max[y~=yi] f(y,i)
% returns z, a 1xn matrix of best wrong answers: argmax[y~=yi] f(y,i)
% for binary models m = y .* f and there is no z
% Yi: 1D indices of correct answers for multi
if (c>1) Yi = y_tr + c*[0:n-1]; end

function [d,z]=margins(f)
if (c == 1)
  d = y_tr .* f;
else
  fYi = f(Yi);                  % save all correct answers
  f(Yi) = -inf;                 % replace them with -inf
  [maxf,z] = max(f);            % find best wrong answers
  d = fYi - maxf;               % margin is the difference
end % if
end % margins

tic(); 
fprintf('Computing initial scores...\n');
initial_scores = scores(x_tr, model, p.average);
initial_test_scores = [];
if n_te initial_test_scores = scores(p.x_te, model, p.average); end
toc();

% computing target margins.  paper assumes svm with
% margin=1 and uses epsilon=0.5.  since our margins are on an
% arbitrary scale we will scale them with mean of positive
% margins.
target_margin = margins(initial_scores);
mean_margin = mean(target_margin(target_margin>0));
target_margin(target_margin<0) = -inf; % to ignore mistakes in target
epsilon_scaled = p.epsilon * mean_margin;
margin_scaled = p.margin * mean_margin;
fprintf('Scaled epsilon = %g, margin = %g\n', epsilon_scaled, margin_scaled);
target_margin(target_margin>margin_scaled) = margin_scaled;

% the paper suggests a learning rate of 0.5 for margin=1 and
% k(x,x)=1.  we will use mean_abs_beta to scale our learning rate.
if (p.average == 0) eta_scaled = p.eta * mean(abs(model.beta(:)));
else eta_scaled = p.eta * mean(abs(model.beta2(:))); end
fprintf('Scaled eta = %g\n', eta_scaled);


% Prepare new model
m = model;
m.iter = 0;  % training iterations
m.S=[];      % 1xs support vector indices
m.SV=[];     % dxs support vectors
m.beta=[];   % cxs support vector weights
m.beta2=[];  % cxs averaged weights (unused)
m.pred=zeros(c,n); % cxn score for each training instance
m.pred_te = []; % c x n_te score for each test instance
if n_te m.pred_te=zeros(c, n_te); end
m.numSV=[];  % tx1 number of SV for each iteration (unused)
m.aer=[];    % tx1 number of errors for each iteration (unused)
m.errTot = 0; % final number of errors (unused)

% Start clock and report original model performance
tic(); 
err_report(initial_scores, initial_test_scores, 0, nsv, 0, 1);

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
    m.SV(:,end+1) = x_tr(:,xi);
    si = numel(m.S);
  end

  if c == 1                     % binary update
    d_beta = y_tr(xi) * eta_scaled;
  else                          % multiclass update
    d_beta = zeros(c, 1);
    d_beta(y_tr(xi)) = eta_scaled;
    d_beta(z(xi)) = -eta_scaled;
  end
  if si > size(m.beta, 2)
    assert(si == 1 + size(m.beta, 2));
    m.beta(:,si) = d_beta;
  else
    m.beta(:,si) = m.beta(:,si) + d_beta;
  end
  m.pred = m.pred + d_beta * feval(m.ker,m.SV(:,si),x_tr,m.kerparam);
  if n_te 
    m.pred_te = m.pred_te + d_beta * feval(m.ker,m.SV(:,si),p.x_te,m.kerparam);
  end
  m.numSV(m.iter) = numel(m.S);
  if mod(m.numSV(m.iter), m.step) == 0 && m.numSV(m.iter) ~= m.numSV(m.iter-1)
    err_report(m.pred, m.pred_te, m.iter, numel(m.S), maxdiff);
  end % if

end % while

err_report(m.pred, m.pred_te, m.iter, numel(m.S), maxdiff);

function err_report(scores_tr, scores_te, iter, nsv, maxdiff, title)
if (nargin < 6) title = 0; end
if title fprintf('time\titer\tnsv\terr_tr\terr_te\tmaxdiff\n'); end
err_tr = numel(find(predicted_labels(scores_tr) ~= y_tr));
err_te = 0;
if n_te err_te = numel(find(predicted_labels(scores_te) ~= p.y_te)); end
fprintf('%d\t%d\t%d\t%d\t%d\t%d\n', round(toc()), iter, nsv, err_tr, ...
        err_te, maxdiff);
end % err_report

end % model_sparsify
