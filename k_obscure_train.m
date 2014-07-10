function model = k_obscure_train(K, Y, model, options)
% K_OBSCURE_ONLINE_TRAIN  OBSCURE Algorithm
%
%    MODEL = K_OBSCURE_ONLINE_TRAIN(K,Y,MODEL) trains an p-norm Multi
%    Kernel classifier according to the Online-Batch Strongly Convex mUlti
%    keRnel lEarning algorithm, using precomputed kernels.
%     
%    Inputs:
%    K -  3-D N*N*F Kernel Matrices, each kernel K(:, :, i) is a N*N matrix
%    Y -  Training labels, N*1 Vector
%
%    Additional parameters:
%    - model.p is 'p' of the p-norm used in the regularization
%      Default value is  1/(1-1/(2*log(number_of_kernels))).
%    - model.T1 is maximum numer of training epochs for the online stage.
%      The online stage will stop earlier if it converges.
%      Default value is 5.
%    - model.T2 is numer of training epochs for the batch stage.
%      Default value is 5.
%    - model.lambda is the regularization weight.
%      Default value is 1/numel(Y).
%    - model.eta
%      Default value is numbers_of_cue^(-2/q).
%
%   References:
%     - Orabona, F., Jie, L., and Caputo, B. (2010).
%       Online-Batch Strongly Convex Multi Kernel Learning.
%       Proceedings of the 23rd IEEE Conference on Computer Vision and
%       Pattern Recognition.

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

if nargin<4
    options = [];
end

if ~isfield(model,'lambda')
  if isfield(model, 'C')==0  
     model.lambda = 1/numel(Y);
  else
     model.lambda = 1/(numel(Y)*model.C);
  end
end

if isfield(model,'T1')
    model.T = model.T1;
else
    model.T = 5;
end
if isfield(model,'S1')
   model.step = model.S1;
end

display('Training OBSCURE stage 1 - online ....')
model_online = k_obscure_online_train(K, Y, model, options);
display('done!')

model_online = rmfield(model_online,'iter');
if isfield(model,'T2')
    model_online.T = model.T2;
else
    model_online.T = 5;
end
if isfield(model,'S2')
   model_online.step = model.S2;
end

display('Training OBSCURE stage 2 - batch ....')
model = k_obscure_batch_train(K, Y, model_online, options);
display('done!')
