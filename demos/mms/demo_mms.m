% Demo for MMS algorithm on USPS dataset.
%
%   References:
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

% demo
clc
clear all
load('../data/usps_libsvm.mat');
load('../data/usps_mms.mat');

model_init = [];
model_init.n_cla = 10;
model_init.step  = 10;

model_init.R = 5;
model_init.T = 100;
model_init.bias = 1;
model_init.proj = 0;

tic
model_mms = mms_multi_train(X, S, Y, model_init, xtest', [], ytest');
toc
