function model = pnorm_train(X,Y,model)
% PNORM_TRAIN p-Norm algorithm
%
%    MODEL = PNORM_TRAIN(X,Y,MODEL) trains a classifier according to the
%    p-Norm algorithm.
%
%    Additional parameters:
%    - model.p is the norm used by the algorithm. It must be bigger than or
%      equal to 2.
%      Default value is 2*log(number of features).
%
%   References:
%     - Gentile, C. (2003).
%       The Robustness of the p-Norm Algorithms.
%       Machine Learning 53, (pp. 265–299).

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
    model.w=spalloc(1,size(X,1),1);
    model.w2=zeros(1,size(X,1));
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(numel(Y),1);
    
    model.theta=spalloc(1,size(X,1),1);
end

if isfield(model,'p')==0
    model.p=max(2*log(size(X,1)),2);
end

model.q=1/(1-1/model.p);

for i=1:n
    model.iter=model.iter+1;
        
    val_f=model.w*X(:,i);

    Yi=Y(i);
    
    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(model.iter)=val_f;
    
    if Yi*val_f<=0
        model.theta=model.theta+Yi*X(:,i)';
        abs_theta=abs(model.theta);
        %model.w=(sign(model.theta).*abs_theta.^(model.p-1))/(eps+norm(abs_theta,model.p)^(model.p-2));
        model.w=model.theta.*(abs_theta/(eps+norm(abs_theta,model.p))).^(model.p-2);
        model.S(end+1)=model.iter;
    end

    model.w2=model.w2+model.w;
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end