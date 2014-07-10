function model = k_perceptron_train(X,Y,model)
% K_PERCEPTRON_TRAIN Kernel Perceptron/Random Budget Perceptron algorithm 
%
%    MODEL = K_PERCEPTRON_TRAIN(X,Y,MODEL) trains an classifier according
%    to the Perceptron/Random Budget Perceptron algorithm, using kernels.
%
%    Additional parameters: 
%    - model.maxSV is the maximum number of Support Vectors. When the
%      algorithm reaches that quantity it starts discarding random vectors,
%      according to the Random Budget Perceptron algorithm.
%      Default value is inf.
%
%   References:
%     - Cavallanti, G., Cesa-Bianchi, N., & Gentile, C. (2007).
%       Tracking the best hyperplane with a simple budget Perceptron
%       Machine Learning, 69(2-3), (pp. 143-167).
%
%     - Rosenblatt, F. (1958).
%       The Perceptron: A probabilistic model for information storage and
%       organization in the brain.
%       Psychological Review, 65, (pp. 386-407).

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
    model.iter=0;
    model.beta=[];
    model.beta2=[];
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(numel(Y),1);
end

if isfield(model,'maxSV')==0
    model.maxSV=inf;
end

for i=1:n
    model.iter=model.iter+1;
        
    if numel(model.S)>0
        if isempty(model.ker)
            K_f=X(model.S,i);
        else
            K_f=feval(model.ker,model.SV,X(:,i),model.kerparam);
        end
        val_f=model.beta*K_f;
    else
        val_f=0;
    end

    Yi=Y(i);
    
    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;

    model.pred(model.iter)=val_f;
    
    if Yi*val_f<=0
        if numel(model.S)==model.maxSV
            mn_idx=ceil(model.maxSV*rand);
            model.beta(mn_idx)=Yi;
            if ~isempty(model.ker)
                model.SV(:,mn_idx)=X(:,i);
            end
            model.S(mn_idx)=model.iter;
        else
            model.beta(end+1)=Yi;
            model.S(end+1)=model.iter;
            if ~isempty(model.ker)
                model.SV(:,end+1)=X(:,i);
            end
        end

        if model.maxSV==inf
            model.beta2(end+1)=0;
        end
    end

    if model.maxSV==inf
        model.beta2=model.beta2+model.beta;
    else
        model.beta2=zeros(size(model.beta));
    end
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end    
end
