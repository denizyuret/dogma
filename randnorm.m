function x = randnorm(n, m, S, V)
% RANDNORM Sample from multivariate normal
%
%   X = RANDNORM(N,M) returns a matrix of N columns where each column is a
%   sample from a multivariate normal with mean M (a column vector) and
%   unit variance.
%
%   X = RANDNORM(N,M,S) specifies the standard deviation, or more generally
%   an upper triangular Cholesky factor of the covariance matrix. This is
%   the most efficient option.
%
%   X = RANDNORM(N,M,[],V) specifies the covariance matrix.

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

if nargin == 1
  x = randn(1,n);
  return;
end
[d,nm] = size(m);
x = randn(d, n);
if nargin > 2
  if nargin == 4
    if d == 1
      S = sqrt(V);
    else
      S = chol(V);
    end
  end
  if d == 1
    x = S .* x;
  else
    x = S' * x;
  end
end
if nm == 1
  x = x + repmat(m, 1, n);
else
  x = x + m;
end
