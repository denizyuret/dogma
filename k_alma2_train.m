function model = k_alma2_train(X,Y,model)
% K_ALMA2_TRAIN Kernel Approximate Maximal Margin Algorithm with the 2-norm
%
%    MODEL = K_ALMA2_TRAIN(X,Y,MODEL) trains a classifier according to the
%    Approximate Maximal Margin Algorithm algorithm, using the 2-norm and
%    kernels.
%
%    Additional parameters:
%    - model.alpha sets the fraction of the margin.
%      Default value is 0.
%    - model.B is the value of the initial threshold for the margin.
%      Default value is 1.
%    - model.C is the value of the initial factor used in the updates.
%      Default value is sqrt(2)-1.
%
%    Note that the default values do not correspond to the one suggested in
%    the original paper of Gentile (2001), but in my experiments they give
%    the best results. These default values satisfies the condition of
%    Theorem 3 in the paper of Gentile (2001).
%
%   References:
%     - Gentile, C. (2001).
%       A New Approximate Maximal Margin Classification Algorithm.
%       Journal of Machine Learning Research 2(Dec), (pp. 213-242).

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
    model.iter     = 0;
    model.beta     = [];
    model.beta2    = [];
    model.errTot   = 0;
    model.errTotAv = 0;
    model.numSV    = zeros(numel(Y),1);
    model.aer      = zeros(numel(Y),1);
    model.aerAv    = zeros(numel(Y),1);
    model.pred     = zeros(numel(Y),1);
    model.pred2    = zeros(numel(Y),1);

    model.norm2W   = 0;
    model.k        = 1;
end

if isfield(model,'alpha')==0
    model.alpha = 0;
end

if isfield(model,'B')==0
     % with B=1 we do not have margins impossible to satisfy, but we
     % maximize the number of margin updates
    model.B = 1;
end

if isfield(model,'C')==0
    model.C = sqrt(2)-1;
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
        val_f2=model.beta2*K_f;
    else
        if isempty(model.ker)
            Kii=X(i,i);
        else
            Kii=feval(model.ker,X(:,i),X(:,i),model.kerparam);
        end
        val_f=0;
        val_f2=0;
    end

    Yi=Y(i);
    
    model.errTot=model.errTot+(sign(val_f)~=Yi);
    model.aer(model.iter)=model.errTot/model.iter;

    model.errTotAv=model.errTotAv+(sign(val_f2)~=Yi);
    model.aerAv(model.iter)=model.errTotAv/model.iter;

    model.pred(model.iter)=val_f;
    model.pred2(model.iter)=val_f2;
    
    if Yi*val_f/sqrt(Kii)<=(1-model.alpha)*model.B/sqrt(model.k);
        eta=model.C/sqrt(model.k);
        
        model.beta(end+1)=Yi*eta/sqrt(Kii);
        model.norm2W=model.norm2W+2*eta*Yi*val_f/sqrt(Kii)+eta^2;
        N=max(1,sqrt(model.norm2W));

        model.beta=model.beta/N;
        model.norm2W=model.norm2W/N^2;
        
        model.S(end+1)=model.iter;
        if ~isempty(model.ker)
            model.SV(:,end+1)=X(:,i);
        end
        
        model.beta2(end+1)=0;
        model.k=model.k+1;
    end

    model.beta2=model.beta2+model.beta;
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
        fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end
end