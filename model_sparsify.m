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

%%% Use gpu if there is one:

if gpuDeviceCount()
  gpudev = gpuDevice();
  fprintf('Using gpu.\n');
  gpu = 1;
else
  gpu = 0;
  max_num_el = 1e8;	% maximum number of doubles during score calc
end

%%% Process input arguments

% Make sure we have a kernel based model
assert(isfield(model,'beta'), 'Need kernel model');
assert(isfield(model,'ker') && ~isempty(model.ker), 'Need kernel model');

% Use averaged hyperplain (beta2) by default
if (nargin < 4) p = []; end
if (~isfield(p,'average')) p.average = 1; end
if (~isfield(model,'beta2') || isempty(model.beta2)) p.average = 0; end
if p.average fprintf('Using averaged solution beta2.\n');
else fprintf('Using the last solution beta.\n'); end

% Set default margin threshold and learning rate
if (~isfield(p, 'epsilon')) p.epsilon = 0.5; end
if (~isfield(p, 'margin')) p.margin = 1.0; end
if (~isfield(p, 'eta')) p.eta = 0.5; end
fprintf('epsilon=%g margin=%g eta=%g\n', p.epsilon, p.margin, p.eta);

%%% Initialize constants and arrays

n_dim = size(x_tr, 1);
n_tr = size(x_tr, 2);
n_cla = size(model.beta, 1);
n_sv = size(model.beta, 2);

if isfield(p, 'x_te')
  x_te = p.x_te;
  y_te = p.y_te;
  n_te = numel(y_te);
else
  n_te = 0;
end

if gpu
  tic(); fprintf('Sending data to gpu...\n');
  x_tr = gpuArray(x_tr);
  y_tr = gpuArray(y_tr);
  if n_te
    x_te = gpuArray(x_te);
    y_te = gpuArray(y_te);
  end
  toc();
end
    
if (n_cla > 1) 
  Yi = y_tr + n_cla*[0:n_tr-1];
end % Yi: 1D indices of correct answers for multi

fprintf('train: %d, test: %d, classes: %d, nsv: %d\n', n_tr, n_te, n_cla, n_sv);

%%% Define some helper functions

% scores calculates the cxn score matrix for each class and each instance
% only called in the beginning for scores of the initial model.  This
% is a reimplementation of model_predict but it doesn't gather the result
% at the end. it doesn't need to: both margins and pred_labels can 
% handle gpuArray.

function f=scores(x,mo,av)

nd = size(x, 1);
nx = size(x, 2);
nc = size(mo.beta, 1);
ns = size(mo.beta, 2);
fprintf('scores: x(%d,%d) beta(%d,%d)\n',nd,nx,nc,ns);

if isfield(model,'X')
  sv = mo(:,model.S);
else
  sv = mo.SV;
end

if ~av
  beta = mo.beta;
  b = mo.b;
else
  beta = mo.beta2;
  b = mo.b2;
end

if gpu
  tic(); fprintf('Sending data to gpu\n');
  f = zeros(nc,nx,'gpuArray');
  sv = gpuArray(sv);
  beta = gpuArray(beta);
  toc(); fprintf('gpudev.FreeMemory=%g\n', gpudev.FreeMemory);
  xstep = floor(0.9*gpudev.FreeMemory/(8*2*ns));
else
  f = zeros(nc,nx);
  xstep = floor(max_num_el/(2*ns));
end

tic(); fprintf('Processing %d chunks of k(%d,%d)...\n', ceil(nx/xstep), ns, xstep);
for i=1:xstep:nx
  j = min(i+xstep-1,nx);
  f(:,i:j) = b + beta * feval(mo.ker, sv, x(:,i:j), mo.kerparam);
end % for
toc();
clear sv beta
end % scores

% predicted_labels takes the cxn score matrix and predicts labels

function l=predicted_labels(f)
  if n_cla > 1                        % multiclass
    [~, l] = max(f, [], 1);
  else                          % binary
    l = sign(f);
  end % if
end % predicted_labels

% margins() takes f, a cxn matrix of scores
% returns d, a 1xn matrix of margins: f(yi,i) - max[y~=yi] f(y,i)
% returns z, a 1xn matrix of best wrong answers: argmax[y~=yi] f(y,i)
% for binary models d = y .* f and there is no z

function [d,z]=margins(f)
if (n_cla == 1)
  d = y_tr .* f;
else
  fYi = f(Yi);                  % save all correct answers
  f(Yi) = -inf;                 % replace them with -inf
  [maxf,z] = max(f);            % find best wrong answers
  d = fYi - maxf;               % margin is the difference
end % if
end % margins

fprintf('Computing initial scores...\n');
initial_scores = scores(x_tr, model, p.average);
if n_te initial_test_scores = scores(p.x_te, model, p.average);
else initial_test_scores = []; end

% computing target margins.  paper assumes svm with
% margin=1 and uses epsilon=0.5.  since our margins are on an
% arbitrary scale we will scale them with mean of positive
% margins.

target_margin = margins(initial_scores);
mean_margin = mean(target_margin(target_margin>0));
if gpu mean_margin = gather(mean_margin); end
epsilon_scaled = p.epsilon * mean_margin;
margin_scaled = p.margin * mean_margin;
fprintf('Scaled epsilon = %g, margin = %g\n', epsilon_scaled, margin_scaled);
target_margin(target_margin<0) = -inf; % to ignore mistakes in target
target_margin(target_margin>margin_scaled) = margin_scaled;

% scale eta: the paper suggests a learning rate of 0.5 for margin=1 and
% k(x,x)=1.  a perceptron would add a +1 or -1 to beta(i).  We will
% use norm of beta to scale.

% compute mean squared beta
if p.average
  msbeta = sum(model.beta2(:).^2)/n_sv;
else
  msbeta = sum(model.beta(:).^2)/n_sv;
end
if (n_cla > 1) msbeta = msbeta / 2; end 	% for multiclass we change two entries
eta_scaled = p.eta * sqrt(msbeta);
fprintf('Scaled eta = %g\n', eta_scaled);

% Prepare new model by rediscovering all support vectors
newS=zeros(1,0);     % new support vector indices
newSV=zeros(n_dim, 0);    % new support vectors
newbeta=zeros(n_cla, 0);  % new support vector weights
pred_tr = zeros(n_cla,n_tr);        % cxn score for each training instance 
pred_te = zeros(n_cla,n_te);        % cxn_te score for each test instance

if gpu
  newSV = gpuArray(newSV);
  pred_tr = gpuArray(pred_tr);
  pred_te = gpuArray(pred_te);
end

% Start clock and report original model performance
tic(); 
err_report(initial_scores, initial_test_scores, 0, n_sv, 0, 1);
n_sv = 0;                        % number of support vectors
iter = 0;			% number of sparsify iterations

% Do the thing
while 1
  iter=iter+1;
  [d,z] = margins(pred_tr);
  mdiff = target_margin - d;

  [maxdiff, si] = max(mdiff(newS)); % Aggressive version:
  if gpu maxdiff = gather(maxdiff); si = gather(si); end
  if maxdiff > epsilon_scaled   % first check existing sv with violation
    xi = newS(si);
  else                          % otherwise check all instances
    [maxdiff, xi] = max(mdiff);
    if gpu maxdiff = gather(maxdiff); xi = gather(xi); end
    if maxdiff < epsilon_scaled
      break                     % quit if no violation
    end
    newS(:,end+1) = xi;        % add new sv if violation
    newSV(:,end+1) = x_tr(:,xi);
    si = numel(newS);
  end

  if n_cla == 1                     % binary update
    d_beta = y_tr(xi) * eta_scaled;
  else                          % multiclass update
    d_beta = zeros(n_cla, 1);
    d_beta(y_tr(xi)) = eta_scaled;
    d_beta(z(xi)) = -eta_scaled;
  end
  if si > size(newbeta, 2)
    assert(si == 1 + size(newbeta, 2));
    newbeta(:,si) = d_beta;
  else
    newbeta(:,si) = newbeta(:,si) + d_beta;
  end
  pred_tr = pred_tr + d_beta * feval(model.ker,newSV(:,si),x_tr,model.kerparam);
  if n_te pred_te = pred_te + d_beta * feval(model.ker,newSV(:,si),x_te,model.kerparam); end
  if ((mod(numel(newS), model.step) == 0) && (numel(newS) ~= n_sv))
    err_report(pred_tr, pred_te, iter, numel(newS), maxdiff);
  end % if
  n_sv = numel(newS);
end % while

err_report(pred_tr, pred_te, iter, numel(newS), maxdiff);
clear pred_tr pred_te
newSV = gather(newSV);

% construct new model
% make new beta a scaled down version of newbeta
% keeping the same norm as old beta at the same number of SV.
% same goes for beta2 if it exists

m = model;
m.S=newS;      % 1xs support vector indices
m.SV=newSV;     % dxs support vectors
assert(n_sv == size(newbeta,2));
assert(n_sv <= size(model.beta,2));
n1 = norm(model.beta(:,1:n_sv), 'fro');
n2 = norm(newbeta, 'fro');
m.beta = newbeta * (n1 / n2);
if isfield(m,'beta2') && ~isempty(m.beta2)
  n1 = norm(model.beta2(:,1:n_sv), 'fro');
  m.beta2 = newbeta * (n1 / n2);
end

% err_report gives periodic updates during training
function err_report(scores_tr, scores_te, iter, nsv, maxdiff, title)
if (nargin < 6) title = 0; end
if title fprintf('time\titer\tnsv\terr_tr\terr_te\tmaxdiff\n'); end
err_tr = numel(find(predicted_labels(scores_tr) ~= y_tr));
if n_te err_te = numel(find(predicted_labels(scores_te) ~= p.y_te)); else err_te = 0; end
fprintf('%d\t%d\t%d\t%d\t%d\t%d\n', round(toc()), iter, nsv, err_tr, err_te, maxdiff);
end % err_report

end % model_sparsify

