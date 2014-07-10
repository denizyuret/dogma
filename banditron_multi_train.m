function model = banditron_multi_train(X,Y,model)
% BANDITRON_MULTI_TRAIN Banditron algorithm
%
%    MODEL = BANDITRON_MULTI_TRAIN(X,Y,MODEL) trains a multiclass
%    classifier according to the Banditron algorithm.
%
%    Additional parameters:
%    - model.n_cla is the number of classes.
%    - model.gamma is the parameter that controls the trade-off between
%      exploration and exploitation.
%      Default value is 0.01.
%
%   References:
%     - Kakade, S. M., Shalev-Shwartz, S., & Tewari, A. (2008).
%       Efficient bandit algorithms for online multiclass prediction.
%       Proceedings of the 25th International Conference on Machine
%       Learning (pp. 440–447).

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

n = length(Y);   % number of training samples

if isfield(model,'iter')==0
    model.iter = 0;
    model.w = zeros(model.n_cla,size(X,1));
    model.errTot = 0;
    model.numSV = zeros(numel(Y),1);
    model.aer = zeros(numel(Y),1);
    model.pred = zeros(model.n_cla,numel(Y));
end

if isfield(model,'gamma')==0
    model.gamma = .01;
end

for i=1:n
    model.iter = model.iter+1;
    
    val_f = model.w*X(:,i);
    
    Yi = Y(i);
    
    [mx_f,y_hat] = max(val_f);
    Prob = zeros(1,model.n_cla)+model.gamma/model.n_cla;
    Prob(y_hat) = Prob(y_hat)+1-model.gamma;
    random_vect = (rand<cumsum(Prob));
    [dummy,y_tilde] = max(random_vect);
    
    model.errTot = model.errTot+(y_tilde~=Yi);
    model.aer(model.iter) = model.errTot/model.iter;
    model.pred(:,model.iter) = val_f;

    model.w(y_hat,:) = model.w(y_hat,:)-X(:,i)';
    
    if y_tilde==Yi
        model.w(Yi,:) = model.w(Yi,:)+1/Prob(y_tilde)*X(:,i)';
    end
    
    model.numSV(model.iter) = numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f AER:%5.2f\n', ...
            ceil(i/1000),model.aer(model.iter)*100);
    end
end
