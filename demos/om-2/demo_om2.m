% Demo for OM-2 algorithm on IDOL2 dataset.
%
%   References:
%     - Jie, L., Orabona, F., Fornoni, M., Caputo, B., and Cesa-Bianchi, N. (2010).
%       OM-2: An Online Mutli-class Multi-kernel Learning Algorithm.
%       Proceedings of the 23rd IEEE Conference on Computer Vision and
%       Pattern Recognition - Workshops.

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

mcmodel_bak.step   = 100;
mcmodel_bak.n_cue  = n_cue; 
mcmodel_bak.n_cla  = max(train_label);

fprintf('Training OM-2 model, 1 pass...\n');
mcmodel = k_om2_multi_train(train_data,train_label,mcmodel_bak);
fprintf('Done!\n');

fprintf('Test OM-2 model, after 1 pass ...\n')
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