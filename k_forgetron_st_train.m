function model = k_forgetron_st_train(X,Y,model)
% K_FORGETRON__ST_TRAIN Kernel Forgetron algorithm, 'self-tuned' variant
%
%    MODEL = K_FORGETRON_TRAIN(X,Y,MODEL) trains an classifier according
%    to the Forgetron algorithm, 'self-tuned' variant, using kernels.
%
%    Additional parameters: 
%    - model.maxSV is the maximum number of Support Vectors. When the
%      algorithm reaches that quantity it starts discarding random vectors,
%      according to the Forgetron algorithm.
%      Default value is 1/10 of the training samples.
%
%   References:
%     - Dekel, O., Shalev-Shwartz, S., & Singer, Y. (2007).
%       The Forgetron: A kernel-based perceptron on a budget.
%       SIAM Journal on Computing 37, (pp. 1342–1372).

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
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(numel(Y),1);

    model.out=[];
    model.Q=0;
end

if isfield(model,'maxSV')==0
    model.maxSV=numel(Y)/10;
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

    model.errTot=model.errTot+(sign(val_f)~=Y(i));
    model.aer(model.iter)=model.errTot/model.iter;
    
    if Y(i)*val_f<=0
        model.beta(end+1)=Y(i);
        model.S(end+1)=model.iter;
        if ~isempty(model.ker)
           model.SV(:,end+1)=X(:,i);
        end
        
        if numel(model.S) > model.maxSV
            if isempty(model.ker)
                K_f=X(model.S,model.S(1));
            else
                K_f=feval(model.ker,model.SV,model.SV(:,1),model.kerparam);
            end
            fp=model.beta*K_f;

            a=model.beta(1)^2-2*model.beta(1)*fp;
            b=2*abs(model.beta(1));
            c=model.Q-15/32*model.errTot;
            d=b^2-4*a*c;
            if a>0 || (a<0 && d>0 && (-b-sqrt(abs(d)))/(2*a)>1)
                phi=min(1,(-b+sqrt(d))/(2*a));
            elseif a==0
                phi=min(1,-c/b);
            else
                phi=1;
            end
            
            model.beta=model.beta*phi;
            
            fpp=model.beta*K_f;
            e=abs(model.beta(1));
            model.Q=model.Q+e^2+2*e-2*e*sign(model.beta(1))*fpp;
            
            model.beta(1)=[];
            model.S(1)=[];
            if ~isempty(model.ker)
                model.SV(:,1)=[];
            end
        end
    end
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
      fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/model.iter*100,numel(model.S),model.aer(model.iter)*100);
    end        
end
