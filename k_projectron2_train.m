function model = k_projectron2_train(X,Y,model)
% K_PROJECTRON_TRAIN Kernel Projectron++ algorithm
%
%    MODEL = K_PROJECTRON_TRAIN(X,Y,MODEL) trains an classifier according
%    to the Projectron++ algorithm, using kernels.
%
%    Additional parameters:
%    - model.eta is the sparseness parameter, used to trade-off the
%      performance for sparseness of the classifier.
%      Default value is 0.1.
%
%   References:
%     - Orabona, F., Keshet, J., & Caputo, B. (2009).
%       Bounded Kernel-Based Online Learning.
%       Journal of Machine Learning Research 10(Nov), (pp. 2643–2666).

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
end

if isfield(model,'eta')==0
    model.eta=.1;
end

n_proj1=0;
n_proj2=0;
n_skip=0;

for i=1:n
    model.iter=model.iter+1;
        
    if numel(model.S)>0
        K_f=feval(model.ker,model.SV,X(:,i),model.kerparam);
        val_f=model.beta*K_f;
    else
        val_f=0;
        K_f=0;
    end
    
    model.errTot=model.errTot+(sign(val_f)~=Y(i));
    model.aer(model.iter)=model.errTot/model.iter;
    
    model.pred(model.iter)=val_f;

    if Y(i)*val_f<1 && Y(i)*val_f>0     % Margin Error
        loss=(1-Y(i)*val_f);
        Kii=feval(model.ker,X(:,i),X(:,i),model.kerparam);
        
        coeff=K_f'*model.Kinv;
        % 'max' to prevent numerical instabilities that could make delta a
        % negative quantity.
        delta=max(Kii-coeff*K_f,0);
        norm_xt=max(Kii-delta,0);     

        if loss-delta/(model.eta)>0
            alpha=min(min(loss/norm_xt,1),2*(loss-delta/(model.eta))/norm_xt);
            model.beta=model.beta+alpha*Y(i)*coeff;
            n_proj2=n_proj2+1;
        else
            n_skip=n_skip+1;
        end
    elseif Y(i)*val_f<=0                % Mistake
        Kii=feval(model.ker,X(:,i),X(:,i),model.kerparam);
        coeff=K_f'*model.Kinv;
        % 'max' to prevent numerical instabilities that could make delta a
        % negative quantity.
        delta=max(Kii-coeff*K_f,0);

        if delta<=model.eta
            model.beta=model.beta+Y(i)*coeff;
            n_proj1=n_proj1+1;
        else
            model.beta(end+1)=Y(i);
            model.S(end+1)=model.iter;
            model.SV(:,end+1)=X(:,i);
        
            model.beta2(end+1)=0;
        
            if numel(model.S)>1
                tmp=[model.Kinv, zeros(numel(model.S)-1,1);zeros(1,numel(model.S))];
                tmp=tmp+[coeff'; -1]*[coeff'; -1]'/delta;
            else
                tmp=feval(model.ker,model.SV,model.SV,model.kerparam)^-1;
            end
            model.Kinv=tmp;
        end
    end
    
    model.beta2=model.beta2+model.beta;
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
        fprintf('#%.0f SV:%5.2f(%d)\tproj:%5.2f\tAER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/i*100,numel(model.S),n_proj1/i*100,model.aer(model.iter)*100);
    end
end
