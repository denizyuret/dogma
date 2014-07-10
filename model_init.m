function model = model_init(kernel,kerparam)
% K_INIT create an empty model for training
%
%   MODEL = K_INIT(KERNEL_FUNCTION,KERNEL_PARAMS) returns an empty model,
%   that will use KERNEL_FUNCTION with KERNEL_PARAMS.
%
%   MODEL = K_INIT returns an empty model, setting the kernel to null. It
%   is used for algorithms without kernels.
%
%   Example:
%       % Define a Gaussian kernel, with scale = 2
%       hp.type='rbf';
%       hp.gamma=2;
%       model_bak = model_init(@compute_kernel,hp);

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
%    Contact the author: francesco [at] orabona.com

if nargin==0 || isempty(kerparam)
    kernel=[];
    kerparam=[];
end

model.S=[];
model.SV=[];
model.ker=kernel;
model.kerparam=kerparam;
model.b=0;
model.b2=0;
model.step=1000;