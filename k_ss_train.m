function model = k_ss_train(X,Y,model)
% K_SS_TRAIN Kernel Selective Sampler
%
%    MODEL = K_SS_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Selective Sampling algorithm, using kernels. The algorithm will query
%    a label only on certain rounds.
%
%    Additional parameters:
%    - model.K is the parameter to tune the query rate.
%      Default value is 1.
%
%   References:
%     - Cavallanti, G., Cesa-Bianchi, N., & Gentile, C. (2011)
%       Learning noisy linear classifiers via adaptive and selective sampling
%       Machine Learning, 83, (pp. 71-102).

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

    model.Kinv=0;
    model.Y_S=[];
    model.N=0;
    model.numQueries=0;
    model.nacr=zeros(numel(Y),1);
end

if isfield(model,'K')==0
    model.K=1;
end

for i=1:n
    model.iter=model.iter+1;
        
    if numel(model.S)>0
        if isempty(model.ker)
            K_f=X(model.S,i);
            Kii=X(i,i);
        else
            K_f=feval(model.ker,model.SV,X(:,i),model.kerparam);
            Kii=feval(model.ker,X(:,i),X(:,i),model.kerparam);
        end
        val_f=model.beta*K_f;
    else
        if isempty(model.ker)
            Kii=X(i,i);
        else
            Kii=feval(model.ker,X(:,i),X(:,i),model.kerparam);
        end
        val_f=0;
        K_f=0;
    end

    Yi=Y(i);
    
    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;
    model.nacr(model.iter)=(model.iter-model.errTot)/model.numQueries;
    
    coeff=K_f'*model.Kinv;
    delta=Kii-coeff*K_f;
    
    val_f=val_f/(delta+1);
    model.pred(model.iter)=val_f;
    
    if val_f^2<=Kii*model.K*log(model.iter)/model.N
        model.numQueries=model.numQueries+1;
        model.S(end+1)=model.iter;
        if ~isempty(model.ker)
            model.SV(:,end+1)=X(:,i);
        end
        model.Y_S(end+1)=Yi;

        if numel(model.S)>1
            tmp=[model.Kinv, zeros(numel(model.S)-1,1);zeros(1,numel(model.S))];
            tmp=tmp+[coeff'; -1]*[coeff'; -1]'/(delta+1);
        else
            tmp=full((Kii+1)^-1);
        end
        model.Kinv=tmp;

        model.beta=model.Y_S*model.Kinv;
        model.N=model.N+1;
    end
    
    model.numSV(model.iter)=numel(model.S);

    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\tQueried Labels:%5.2f(%d)\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),...
            model.aer(model.iter)*100,model.numQueries/model.iter*100,...
            model.numQueries);
    end
end