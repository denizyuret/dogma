function [m,n] = model_sparsify(model,x_tr,y_tr,p)

% MODEL_SPARSIFY implements an algorithm that reduces the number of
% support vectors in a kernel based model.  Handles both binary and
% multiclass models.  Uses the averaged solution (beta2) if available
% by default.  Does not handle pre-computed kernel matrices.
%
%    m = model_sparsify(model,x_tr,y_tr,p) takes a model, a set of
%    instances to choose new support vectors from (x_tr,y_tr), and
%    some optional parameters (p) and returns m, a sparser model,
%    reporting its train/test set accuracy.  It does not modify the
%    input model.  It works for any kernel based model, binary or
%    multi-class.
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
%    - p.eta is the learning rate.  Default is 0.3.  It will be
%    multiplied by mean(abs(beta(:))) for scaling and determines
%    the increments applied to m.beta.
%
%    - p.epsilon determines the difference acceptable between the
%    original margins and the new margins.  Default is 0.1.
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

%%% Process input arguments

% Make sure we have a kernel based model
assert(isfield(model,'beta'), 'Need kernel model');
assert(isfield(model,'ker') && ~isempty(model.ker), 'Need kernel model');

% Set some constants
n = struct('cla', size(model.beta, 1), 'sv', size(model.beta, 2),...
           'dim', size(x_tr, 1), 'tr', size(x_tr, 2), 'te', 0);
if isfield(p, 'x_te') n.te = size(p.x_te, 2); end
fprintf('train: %d, test: %d, classes: %d, dims: %d, nsv: %d\n', ...
        n.tr, n.te, n.cla, n.dim, n.sv);

% Use averaged hyperplain (beta2) by default
if (nargin < 4) p = struct(); end
if (~isfield(p,'average')) p.average = 1; end
if (~isfield(model,'beta2') || isempty(model.beta2)) p.average = 0; end
if p.average fprintf('Using averaged solution beta2.\n');
else fprintf('Using the last solution beta.\n'); end

% Set default margin threshold and learning rate
if (~isfield(p, 'epsilon')) p.epsilon = 0.1; end
if (~isfield(p, 'margin')) p.margin = 1.0; end
if (~isfield(p, 'eta')) p.eta = 0.3; end
fprintf('epsilon=%g margin=%g eta=%g\n', p.epsilon, p.margin, p.eta);

% Use gpu if we have one and subsample x_tr if it is too big.
gpu = gpuDeviceCount();
if gpu 
  gpudev = gpuDevice();
  check_tr_size();
  % Be memory efficient if we are going to use the gpu
  assert(max(abs(y_tr)) < 127);
  x_tr = single(x_tr);
  y_tr = int8(y_tr);
  if n.te 
    p.x_te = single(p.x_te); 
    p.y_te = int8(p.y_te);
  end
end

fprintf('Computing initial scores...\n');
init_tr = scores(x_tr, model, p.average);
if n.te init_te = scores(p.x_te, model, p.average);
else init_te = []; end

% computing target margins.  paper assumes svm with
% margin=1 and uses epsilon=0.5.  since our margins are on an
% arbitrary scale we will scale them with mean of positive
% margins.

if (n.cla > 1) y_linear_indices = int32(y_tr) + n.cla*int32(0:n.tr-1); end
target_margin = margins(init_tr);
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
  msbeta = sum(model.beta2(:).^2)/n.sv;
else
  msbeta = sum(model.beta(:).^2)/n.sv;
end
if (n.cla > 1) msbeta = msbeta / 2; end 	% for multiclass we change two entries
eta_scaled = p.eta * sqrt(msbeta);
fprintf('Scaled eta = %g\n', eta_scaled);

% report original model performance
err_report(init_tr, init_te, 0, n.sv, 0, 1);
clear init_tr init_te;

% Prepare new model by rediscovering all support vectors
m = model;
m.S=zeros(1,0);		% new support vector indices - put in gpu if mdiff(newS) slow
m.SV=zeros(n.dim, 0);   % new support vectors
m.beta=zeros(n.cla, 0); % new support vector weights
pred_tr=zeros(n.cla,n.tr); % score for each training instance 
if n.te pred_te=zeros(n.cla,n.te); end % score for each test instance

if gpu
  fprintf('Resetting gpu...\n');
  target_margin = gather(target_margin);
  tic(); reset(gpudev); toc();
  fprintf('Sending data to gpu...\n');
  tic();
  x_tr = gpuArray(single(x_tr));
  pred_tr = gpuArray(single(pred_tr));
  if n.te 
    p.x_te = gpuArray(single(p.x_te));
    pred_te = gpuArray(single(pred_te)); 
  end
  target_margin = gpuArray(single(target_margin));
  y_linear_indices = gpuArray(int32(y_linear_indices));
  toc();
end

% Start clock
tic(); 
n.sv = 0;                               % number of support vectors
n.iter = 0;			        % number of sparsify iterations

% Do the thing
while 1
  n.iter = n.iter+1;
  [pred_margin, pred_y] = margins(pred_tr);
  mdiff = target_margin - pred_margin;

  [maxdiff, si] = max(mdiff(m.S)); % Aggressive version:
  if gpu maxdiff = gather(maxdiff); si = gather(si); end
  if maxdiff > epsilon_scaled   % first check existing sv with violation
    xi = m.S(si);
  else                          % otherwise check all instances
    [maxdiff, xi] = max(mdiff);
    if gpu maxdiff = gather(maxdiff); xi = gather(xi); end
    if maxdiff < epsilon_scaled
      break                     % quit if no violation
    end
    m.S(:,end+1) = xi;        % add new sv if violation
    if gpu m.SV(:,end+1) = gather(x_tr(:,xi));
    else m.SV(:,end+1) = x_tr(:,xi); end
    si = numel(m.S);
  end

  if n.cla == 1                     % binary update
    d_beta = y_tr(xi) * eta_scaled;
  else                          % multiclass update
    d_beta = zeros(n.cla, 1);
    d_beta(y_tr(xi)) = eta_scaled;
    d_beta(pred_y(xi)) = -eta_scaled;
  end
  if si > size(m.beta, 2)
    assert(si == 1 + size(m.beta, 2));
    m.beta(:,si) = d_beta;
  else
    m.beta(:,si) = m.beta(:,si) + d_beta;
  end

  pred_tr = pred_tr + d_beta * feval(model.ker, m.SV(:,si), x_tr, model.kerparam);
  if n.te pred_te = pred_te + d_beta * feval(model.ker, m.SV(:,si), p.x_te, model.kerparam); end

  if ((mod(numel(m.S), model.step) == 0) && (numel(m.S) ~= n.sv))
    err_report(pred_tr, pred_te, n.iter, numel(m.S), maxdiff);
  end % if
  n.sv = numel(m.S);
end % while

err_report(pred_tr, pred_te, n.iter, numel(m.S), maxdiff);
clear pred_tr pred_te

% make new beta a scaled down version of m.beta
% keeping the same norm as old beta at the same number of SV.
% same goes for beta2 if it exists

assert(n.sv == size(m.beta,2));
assert(n.sv <= size(model.beta,2));
n1 = norm(model.beta(:,1:n.sv), 'fro');
n2 = norm(m.beta, 'fro');
m.beta = m.beta * (n1 / n2);
if isfield(m,'beta2') && ~isempty(m.beta2)
  n1 = norm(model.beta2(:,1:n.sv), 'fro');
  n2 = norm(m.beta, 'fro');
  m.beta2 = m.beta * (n1 / n2);
end

%%% Define some helper functions

% scores calculates the cxn score matrix for each class and each instance.
% Only called in the beginning for scores of the initial model.  This
% is a reimplementation of model_predict but it doesn't gather the result
% at the end when using gpu. it doesn't need to: both margins and pred_labels can 
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
  f = zeros(nc,nx,'single','gpuArray');
  sv = gpuArray(single(sv));
  beta = gpuArray(single(beta));
  toc(); fprintf('gpudev.FreeMemory=%g\n', gpudev.FreeMemory);
  xstep = floor(gpudev.FreeMemory/(4*(nd+2*ns)));
else
  f = zeros(nc,nx);
  max_num_el = 1e10;
  xstep = floor(max_num_el/(2*ns));
end

tic(); fprintf('Processing %d chunks of k(%d,%d)\n', ceil(nx/xstep), ns, xstep);
for i=1:xstep:nx
  j = min(i+xstep-1,nx);
  f(:,i:j) = b + beta * feval(mo.ker, sv, x(:,i:j), mo.kerparam);
  fprintf('.');
end % for
fprintf('\n');
toc();
clear sv beta
end % scores

% predicted_labels takes the cxn score matrix and predicts labels

function l=predicted_labels(f)
  if n.cla > 1                        % multiclass
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
if (n.cla == 1)
  d = y_tr .* f;
else
  fYi = f(y_linear_indices);            % save all correct answers
  f(y_linear_indices) = -inf;           % replace them with -inf
  [maxf,z] = max(gather(f));            % find best wrong answers
  z = int8(z);                          % original z is a double
  d = fYi - maxf;                       % margin is the difference
  clear fYi maxf;
end % if
end % margins

% We need space for single arrays of: x_tr(d,n), pred_tr(c,n), 
% pred_margin(1,n), pred_y(1,n), target_margin(1,n), mdiff(1,n), 
% x_te(d,m), pred_te(c,m), y_linear_indices(1,n), 
% fYi(1,n), maxf(1,n), z(1,n)x2

function check_tr_size()
fprintf('Resetting gpu...\n');
tic(); reset(gpudev); toc();
fprintf('GPU memory: %d\n', gpudev.FreeMemory);
max_tr_size = gpudev.FreeMemory/4;
max_tr_size = max_tr_size - n.te * (n.dim + n.cla); % space for x_te, pred_te
assert(max_tr_size > 0);
max_tr_size = floor(max_tr_size / (n.dim + n.cla + 20)); % max n left
if n.tr > max_tr_size 
  tic();
  fprintf('Subsampling x_tr from %d to %d instances to fit in gpu.\n', n.tr, max_tr_size);
  % We should take all SV and pick the rest randomly
  sample = 1:n.tr;
  sorted = sort(model.S);
  for i=1:numel(sorted);
    s = sorted(i);
    assert(sample(s) == s);
    sample(s) = sample(i);
    sample(i) = s;
  end
  for i=numel(sorted)+1:max_tr_size
    r = randi([i max_tr_size]);
    si = sample(i);
    sample(i) = sample(r);
    sample(r) = si;
  end
  sample = sample(1:max_tr_size);
  x_tr = x_tr(:, sample);
  y_tr = y_tr(sample);
  n.tr = size(x_tr, 2);
  assert(n.tr == max_tr_size);
  toc();
end % if
end % check_tr_size

% err_report gives periodic updates during training
function err_report(scores_tr, scores_te, iter, nsv, maxdiff, title)
if (nargin < 6) title = 0; end
if title fprintf('time\titer\tnsv\terr_tr\terr_te\tmaxdiff\n'); end
err_tr = numel(find(predicted_labels(scores_tr) ~= y_tr));
if n.te err_te = numel(find(predicted_labels(scores_te) ~= p.y_te)); else err_te = 0; end
fprintf('%d\t%d\t%d\t%d\t%d\t%g\n', round(toc()), iter, nsv, err_tr, err_te, maxdiff);
end % err_report

end % model_sparsify

