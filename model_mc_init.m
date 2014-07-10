function model = model_mc_init(hp1, hp2)
% MODEL_MC_INIT create an empty multicues model for training
%
%   MODEL = MODEL_MC_INIT(KERNEL_PARAMS_LAYER1, PARAMS_LAYER2) returns an
%   empty model, to be used with multiple cues learning.

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

model.n_cue=numel(hp1);

% initial empty models
% 1st layer
for i=1:numel(hp1)
    model.L1{i, 1} = model_init(@compute_kernel,hp1{i});
end

% 2nd layer
model.L2 = model_init(@compute_kernel,hp2);