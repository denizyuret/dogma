% Demo for UFO-MKL algorithm on Oxford Flower dataset.
% Please read the README first and download the data from the dataset
% website.
%
%   References:
%     - Orabona, F., Jie, L. (2011).
%       Ultra-Fast Optimization Algorithm for Sparse Multi Kernel Learning.
%       Proceedings of the 28th International Conference on Machine Learning.

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

clear
close all
rand('state', 0);

% Load the datasplit and distance matrix
load ../data/datasplits.mat
load ../data/distancematrices17gcfeat06
load ../data/distancematrices17itfeat08.mat

% Create the training data and testing data
% In the demo we use split 1 of the flower dataset
trsplit = trn1;
tesplit = tst1;
Ktrain = zeros(680, 680, 7);
Ktrain(:,:,1)=exp(-D_colourgc(trsplit, trsplit)/mean(mean(D_colourgc(trsplit, trsplit))));
Ktrain(:,:,2)=exp(-D_hog(trsplit, trsplit)/mean(mean(D_hog(trsplit, trsplit))));
Ktrain(:,:,3)=exp(-D_hsv(trsplit, trsplit)/mean(mean(D_hsv(trsplit, trsplit))));
Ktrain(:,:,4)=exp(-D_shapegc(trsplit, trsplit)/mean(mean(D_shapegc(trsplit, trsplit))));
Ktrain(:,:,5)=exp(-D_siftbdy(trsplit, trsplit)/mean(mean(D_siftbdy(trsplit, trsplit))));
Ktrain(:,:,6)=exp(-D_siftint(trsplit, trsplit)/mean(mean(D_siftint(trsplit, trsplit))));
Ktrain(:,:,7)=exp(-D_texturegc(trsplit, trsplit)/mean(mean(D_texturegc(trsplit, trsplit))));
Ktrain = single(Ktrain);

Ktest = zeros(680, 340, 7);
Ktest(:,:,1)=exp(-D_colourgc(trsplit, tesplit)/mean(mean(D_colourgc(trsplit, trsplit))));
Ktest(:,:,2)=exp(-D_hog(trsplit, tesplit)/mean(mean(D_hog(trsplit, trsplit))));
Ktest(:,:,3)=exp(-D_hsv(trsplit, tesplit)/mean(mean(D_hsv(trsplit, trsplit))));
Ktest(:,:,4)=exp(-D_shapegc(trsplit, tesplit)/mean(mean(D_shapegc(trsplit, trsplit))));
Ktest(:,:,5)=exp(-D_siftbdy(trsplit, tesplit)/mean(mean(D_siftbdy(trsplit, trsplit))));
Ktest(:,:,6)=exp(-D_siftint(trsplit, tesplit)/mean(mean(D_siftint(trsplit, trsplit))));
Ktest(:,:,7)=exp(-D_texturegc(trsplit, tesplit)/mean(mean(D_texturegc(trsplit, trsplit))));
Ktest = single(Ktest);

% gernate the labels of the dataset
Ytrain = zeros(1,680);
Ytest  = zeros(1,340);
for i=1:17, 
  Ytrain(1,(i-1)*40+1:(i-1)*40+40) = i;
  Ytest(1,(i-1)*20+1:(i-1)*20+20)  = i;
end
  
% Parameters for UFO-MKL
C                  = 10;
model_zero         = model_init();
model_zero.n_cla   = 17;
model_zero.T       = 300;   % Number of epochs
model_zero.lambda  = 1/(C*numel(Ytrain));

model_zero.step   = 10*numel(Ytrain);
options.eachRound = @ufomkl_test;
options.Ktest     = Ktest;
options.Ytest     = Ytest;

model = k_ufomkl_multi_train(Ktrain, Ytrain, model_zero, options);

semilogx(model.test,model.obj,'x-')
xlabel('Number of updates')
ylabel('Objective function')
grid