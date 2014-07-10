function model = pa_train(X,Y,model)
% PA_TRAIN Passive-Aggressive algorithm
%
%    MODEL = PA_TRAIN(X,Y,MODEL) trains a classifier according to the
%    Passive-Aggressive algorithm, PA-I and PA-II variants.
%
%    Additional parameters:
%    - model.C is the aggressiveness parameter, used to trade-off the loss
%      on the current sample with the update on the current hyperplane.
%      Default value is 1.
%    - model.update is the used to select the update rule. A value of 1
%      selectes PA-I, 2 selects PA-II.
%      Default value is 1.
%
%   References:
%     - Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., & Singer, Y. (2006).
%       Online Passive-Aggressive Algorithms.
%       Journal of Machine Learning Research 7(Mar), (pp. 551-585).

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
    model.w=zeros(1,size(X,1));
    model.w2=zeros(1,size(X,1));
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(numel(Y),1);
end

if isfield(model,'update')==0
    model.update=1; %default update using PA-I
end

if isfield(model,'C')==0
    model.C=1;
end

for i=1:n
    model.iter=model.iter+1;
    
    val_f=model.w*X(:,i);

    Yi=Y(i);
    
    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(model.iter)=val_f;
    
    if Yi*val_f<1
        norm_x_square=norm(X(:,i))^2;
        if model.update==1
            new_beta=min((1-Yi*val_f)/norm_x_square,model.C);
        else
            new_beta=(1-Yi*val_f)/(norm_x_square+1/(2*model.C));
        end
        model.w=model.w+new_beta*Yi*X(:,i)';
        model.S(end+1)=model.iter;
    end

    model.w2=model.w2+model.w;

    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end
