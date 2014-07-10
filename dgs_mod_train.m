function model = dgs_mod_train(X,Y,model)
% DGS_MOD_TRAIN Modified Dekel-Gentile-Sridharan selective sampler algorithm
%
%    MODEL = DGS_MOD_TRAIN(X,Y,MODEL) trains an classifier according to the
%    modified Dekel-Gentile-Sridharan selective sampler algorithm. The
%    algorithm will query a label only on certain rounds.
%
%    Additional parameters:
%    - model.delta is probability coefficient.
%      Default value is 0.1.
%    - model.originalQueryRule if set to 1 it will the original query rule
%      proposed by Dekel et al.
%      Default value is 0.
%    - model.alpha is the constant used in the modified query rule.
%      Default value is 1.
%
%   References:
%     - Orabona, F., Cesa-Bianchi, N. (2011).
%       Better Selective Sampling Algorithms.
%       Proceedings of the 26th International Conference on Machine Learning.
%
%     - Dekel, O., Gentile, C., & Sridharan, K. (2010).
%       Robust Selective Sampling from Single and Multiple Teachers.
%       Proceedings of the 23rd Annual Conference on Learning Theory.

%    This file is part of the DOGMA library for Matlab.
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

model.a=1;

if isfield(model,'iter')==0
    model.iter=0;
    model.w=zeros(1,size(X,1));
    model.w2=zeros(1,size(X,1));
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(numel(Y),1);

    model.KbInv=eye(size(X,1))/model.a;
    model.sum_rt=0;
    model.numAskedLabels=zeros(numel(Y),1);
    model.numQueries=0;
    model.th=zeros(numel(Y),1);
end

if isfield(model,'delta')==0
    model.delta=.1;
end

if isfield(model,'originalQueryRule')==0
    model.originalQueryRule=0;
end

if isfield(model,'alpha')==0
    model.alpha=1;
end


for i=1:n
    model.iter=model.iter+1;

    val_f=model.w*X(:,i);

    Yi=Y(i);
    
    KbInv_x=model.KbInv*X(:,i);
    r=X(:,i)'*KbInv_x;

    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(model.iter)=val_f;
    if model.originalQueryRule==0
       th=model.alpha*r*(4*model.sum_rt+36*log(model.iter/model.delta))*log(model.iter);
    else
       th=r*(1+4*model.sum_rt+36*log(model.iter/model.delta));
    end
    model.th(model.iter)=th;
                
    if val_f^2<=th
        model.numQueries=model.numQueries+1;
        model.S(end+1)=model.iter;
        
        model.w=model.w-sign(val_f)*KbInv_x'/r*max(abs(val_f)-1,0);
        val_f=model.w*X(:,i);
        model.w=model.w+(Yi-val_f)/(1+r)*KbInv_x';

        model.KbInv=model.KbInv-KbInv_x*KbInv_x'/(1+r);
        model.sum_rt=model.sum_rt+r/(r+1);
    end
    
    model.numAskedLabels(model.iter)=model.numQueries;
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\tAskedLabels:%5.2f(%d)\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),...
            model.aer(model.iter)*100,model.numQueries/model.iter*100,...
            model.numQueries);
    end
end