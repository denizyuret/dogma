function K = compute_kernel(in1,in2,in3,in4)
% COMPUTE_KERNEL Calculates the kernel values
%
%   K = COMPUTE_KERNEL(X,IND1,IND2,KERNEL_PARAMS) calculates the kernel
%   matrix for the data X, using the indexes in IND1 and IND2. The type of
%   kernel and the parameters are passed through the structure
%   KERNEL_PARAMS.
%   K is a matrix with a number of rows equal to the number of indexes of
%   IND1 and number of rows equal to the number of indexes in IND2.
%
%   K = COMPUTE_KERNEL(X1,X2,KERNEL_PARAMS) calculates the kernel matrix
%   between the data X1 and the data X2. The type of kernel
%   and the parameters are passed through the structure KERNEL_PARAMS.
%   K is a matrix with a number of rows equal to the number of columns of
%   X1 and number of rows equal to the number of columns of X2.
%
%   KERNEL_PARAMS is a struct with the field type with one of the following
%   strings:
%
%       'linear'              Linear kernel or dot product
%       'poly'                Polynomial kernel
%       'rbf'                 Gaussian Radial Basis Function kernel
%       'sigmoid'             Sigmoidal kernel
%       'triangular'          Triangular kernel
%       'intersection'        Intersection kernel
%       'intersection_sparse' Intersection kernel for sparse matrices
%       'expchi2'             Exponential Chi^2 kernel
%       'expchi2_sparse'      Exponential Chi^2 kernel for sparse matrices
%       'vovk_inf_poly'       Vovk infinite polynomial
%
%   The other fields of KERNEL_PARAMS depend on the specific kernel.
%
%   Note that K is always returned as a dense matrix.
%
%   Warning: For expchi2_sparse each feature vector must have sum of the
%   elements equal to 1.
%
%   Example:
%       % Load the data
%       load synth_2d
%       % Define a Gaussian kernel, with scale = 2
%       hp.type='rbf'
%       hp.gamma=2
%       % Builds the kernel matrix
%       K = compute_kernel(p,1:10,11:20,hp);
%
%   References:
%     - Cristianini, N., Shawe-Taylor, J An Introduction to Support
%       Vector Machines, Cambridge University Press, Cambridge, UK. 2000.

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

if nargin==4
  ind1=in2;
  ind2=in3;
  hp=in4;
  X1=in1(:,ind1);
  X2=in1(:,ind2);
else
  X1=in1;
  X2=in2;
  hp=in3;
end

if size(X1,2)==0
    K = [];
    return;
end

switch hp.type
   case 'linear'
    K = full(X1'*X2);
    
   case 'poly'
    K = (hp.gamma*full(X1'*X2)+hp.coef0).^hp.degree;
      
   case 'dist'
    normX = full(sum(X1.^2,1));
    normY = full(sum(X2.^2,1));
    K = repmat(normX' ,1,size(X2,2)) + ...
                           repmat(normY,size(X1,2),1) - ...
                           2*full(X1'*X2);
    
   case 'rbf'
    normX = full(sum(X1.^2,1));
    normY = full(sum(X2.^2,1));
    K = exp(-hp.gamma*(repmat(normX' ,1,size(X2,2)) + ...
                           repmat(normY,size(X1,2),1) - ...
                           2*full(X1'*X2)));

   case 'rbf_bias'
    normX = full(sum(X1.^2,1));
    normY = full(sum(X2.^2,1));
    K = exp(-hp.gamma*(repmat(normX' ,1,size(X2,2)) + ...
                           repmat(normY,size(X1,2),1) - ...
                           2*full(X1'*X2)))+hp.coef0;
                           
   case 'triangular'
    normX = full(sum(X1.^2,1));
    normY = full(sum(X2.^2,1));
    K = -sqrt(repmat(normX',1,size(X2,2)) + repmat(normY,size(X1,2),1) - 2*full(X1'*X2));

   case 'sigmoid'
    K = tanh(hp.gamma*full(X1'*X2)+hp.coef0);
    
   case 'expchi2'
    K=zeros(size(X1,2),size(X2,2));
    for i=1:size(X1,2)
        h1=X1(:,i);
        for j=1:size(X2,2)
           h2=X2(:,j);
           tmp=sum(((h1-h2).^2)./(h1+h2+eps));
           K(i,j) = exp( -hp.gamma*tmp );
       end
    end

   case 'expchi2_sparse'
    K=zeros(size(X1,2),size(X2,2));
    for i=1:size(X1,2)
        h1=X1(:,i);
        for j=1:size(X2,2)
           h2=X2(:,j);
           tmp=chisquare_sparse(h1,h2);
           K(i,j) = exp( -hp.gamma*tmp );
       end
    end

   case 'intersection'
    K=zeros(size(X1,2),size(X2,2));
    for i=1:size(X1,2)
        h1=X1(:,i);
        for j=1:size(X2,2)
           h2=X2(:,j);
           K(i,j) = sum(min(h1,h2));
        end
    end

   case 'intersection_sparse'
    K=zeros(size(X1,2),size(X2,2));
    for i=1:size(X1,2)
        h1=X1(:,i);
        for j=1:size(X2,2)
           h2=X2(:,j);
           K(i,j) = hist_intersection_sparse(h1,h2);
        end
    end

   case 'vovk_inf_poly'
    K = full(1./(1-hp.gamma*X1'*X2));
    
   case 'same'
    K=zeros(size(X1,2),size(X2,2));
    for i=1:size(X1,2)
        for j=1:size(X2,2)
           K(i,j) = (X1(:,i)==X2(:,j));
       end
    end
    
   otherwise
    error('Unknown kernel');
end
