function model = bbq_train(X,Y,model)
% BBQ_TRAIN Bound on Bias Query Algorithm
%
%    MODEL = BBQ_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Bound on Bias Query Algorithm. The algorithm will query a label only
%    on certain rounds.
%
%    Additional parameters:
%    - model.k is exponent of the query rate.
%      Default value is 1/2.
%
%   References:
%     - Cesa-Bianchi, N., Gentile, C., & Orabona, F. (2009)
%       Robust Bounds for Classiﬁcation via Selective Sampling.
%       Proceedings of the 26th International Conference on Machine
%       Learning.

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
    model.Y_S=[];
    model.N=0;
    model.numAskedLabels=zeros(numel(Y),1);
    model.numQueries=0;
    model.nacr=zeros(numel(Y),1);
end

if isfield(model,'k')==0
    model.k=1/2;
end

for i=1:n
    model.iter=model.iter+1;

    val_f=model.w*X(:,i);

    Yi=Y(i);
    
    KbInv_x=model.KbInv*X(:,i);
    v=X(:,i)'*KbInv_x;

    rt=v/(v+1);

    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(model.iter)=val_f;
                
    %if rt>model.N^-model.k
    if rt>0.5*model.iter^(-model.k)
        model.numQueries=model.numQueries+1;
        model.S(end+1)=model.iter;
        
        model.w=model.w+(Yi-val_f)/(1+v)*KbInv_x';
        model.KbInv=model.KbInv-KbInv_x*KbInv_x'/(1+v);
    else
        model.N=model.N+1;
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