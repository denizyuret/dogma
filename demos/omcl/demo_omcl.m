% Demo for OMCL algorithm on IDOL2 dataset.
%
%   References:
%     - Jie, L., Orabona, F., & Caputo, B. (2009).
%       An online framework for learning novel concepts over multiple cues.
%       Proceeding of the 9th Asian Conference on Computer Vision
%     - Orabona, F., Keshet, J., & Caputo, B. (2009).
%       Bounded Kernel-Based Online Learning.
%       Journal of Machine Learning Research 10(Nov), (pp. 2643–2666).

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
%    Contact the authors: francesco [at] orabona.com
%                         jluo      [at] idiap.ch

% demo
clc
clear
load ../data/idol2.mat

% cues: 1. color; 2.crfh; 3.bows; 4.laser;
n_cue  = 4;
kernel = { 'expchi2_sparse' 'expchi2_sparse' 'expchi2_sparse' 'rbf' }; 
gamma  = [ 10        100       10        0.67  ];
% projectron parameter
eta    = [ 0.5       0.5       0.5       0.5 ];

for i=1:n_cue
    hp1{i}.type  = kernel{i};
    hp1{i}.gamma = gamma(i); 
end
hp2=[];

mcmodel_bak = model_mc_init(hp1, hp2);
for i=1:n_cue
    mcmodel_bak.L1{i}.eta = eta(i);
end
mcmodel_bak.L2.C = 0.1;

mcmodel_bak.step   = 100;
mcmodel_bak.n_cue  = n_cue; 
mcmodel_bak.n_cla  = max(train_label);

fprintf('Training OMCL model ...\n');
mcmodel = k_omcl_multi_train(train_data,train_label,mcmodel_bak);
fprintf('Done!\n');

fprintf('Test Projectron++ models on each cue ...\n');
for i=1:n_cue
    pred = model_predict(test_data{i},mcmodel.L1{i},1);
    err=numel(find(pred~=test_label))/numel(pred);

    err_cls = zeros(max(test_label),1);
    for c=1:max(test_label)
        idx = find(test_label == c);
        err_cls(c) = numel(find(pred(idx) ~= test_label(idx)))/numel(idx);
    end
    err_cls = mean(err_cls);
  
    fprintf('Cues %d - ERR:%5.2f   AVE ERR PER CATE:%5.2f\n', ...
        i, err*100, err_cls*100);   
end
fprintf('Done!\n');

fprintf('Test OMCL model ...\n')
pred = model_predict(test_data,mcmodel);
err=zeros(1, 2);
% average error rate (sample)
err(1)=numel(find(pred~=test_label))/numel(pred);
% average error rate (class)
err_cls = zeros(max(test_label),1);
for i=1:max(test_label)
    idx = find(test_label == i);
    err_cls(i) = numel(find(pred(idx) ~= test_label(idx)))/numel(idx);
end
err(2) = mean(err_cls);
fprintf('          ERR:%5.2f   AVE ERR PER CATE:%5.2f\n', err(1)*100, err(2)*100);
fprintf('Done!\n');