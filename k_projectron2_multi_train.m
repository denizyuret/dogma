function model = k_projectron2_multi_train(X,Y,model)
% K_PROJECTRON2_MULTI_TRAIN Kernel Projectron++ multiclass algorithm
%
%    MODEL = K_PROJECTRON2_MULTI_TRAIN(X,Y,MODEL) trains an classifier
%    according to the Projectron++ multiclass algorithm, using kernels.
%
%    Additional parameters:
%    - model.eta is the sparseness parameter, used to trade-off the
%      performance for sparseness of the classifier. Note that model.eta is
%      the maximum error on EACH single projection; each projected update
%      has 2 projections.
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

if isfield(model,'n_cla')==0
    model.n_cla=max(Y);
end

if isfield(model,'iter')==0
    model.iter=0;
    model.beta=[];
    model.beta2=[];
    model.errTot=0;
    model.numSV=zeros(numel(Y),1);
    model.aer=zeros(numel(Y),1);
    model.pred=zeros(model.n_cla,numel(Y));
    
    for i=1:model.n_cla
        model.Kinv{i}=[];
        model.Y_cla{i}=[];
    end
end

if isfield(model,'eta')==0
    model.eta=.1;
end

n_skip=0;
n_proj1=0;
n_proj2=0;
n_pred=0;
idx_true=[];
idx_wrong=[];

for i=1:n
    model.iter=model.iter+1;
        
    if numel(model.S)>0
        K_f=feval(model.ker,model.SV,X(:,i),model.kerparam);
        val_f=full(model.beta*K_f);
    else
        val_f=zeros(1,model.n_cla);
        K_f=[];
    end

    Yi=Y(i);
    
    tmp=val_f; tmp(Yi)=-inf;
    [mx_val,idx_mx_val]=max(tmp);

    model.errTot=model.errTot+(val_f(Yi)<=mx_val);
    model.aer(model.iter)=model.errTot/model.iter;
    model.pred(:,model.iter)=val_f;
    
    if val_f(Yi) < mx_val+1 %Margin error or mistake
        Kii=full(feval(model.ker,X(:,i),X(:,i),model.kerparam));
         
        delta_true=Kii;
        delta_wrong=Kii;
        if numel(model.S)>0
            idx_true=model.Y_cla{Yi};
            idx_wrong=model.Y_cla{idx_mx_val};

            if numel(idx_true)>0
                coeff_true=K_f(idx_true)'*model.Kinv{Yi};
                % 'max' to prevent numerical instabilities that could make
                % delta a negative quantity.
                delta_true=max(Kii-coeff_true*K_f(idx_true),0);
            end

            if numel(idx_wrong)>0
                coeff_wrong=K_f(idx_wrong)'*model.Kinv{idx_mx_val};
                % 'max' to prevent numerical instabilities that could make
                % delta a negative quantity.
                delta_wrong=max(Kii-coeff_wrong*K_f(idx_wrong),0);
            end
        end

        if val_f(Yi)>mx_val % Margin error
            loss=1-val_f(Yi)+mx_val;
            delta=delta_wrong+delta_true;
            % 2*model.eta because eta is the tollerance on each single
            % projection.
            if loss-delta/(2*model.eta)>0
                tau_m=min(min(loss/(2*Kii-delta),1),2*(loss-delta/(2*model.eta))/(2*Kii-delta));
                if numel(idx_true)>0
                    model.beta(Yi,idx_true)=model.beta(Yi,idx_true)+tau_m*coeff_true;
                end
                if numel(idx_wrong)>0
                    model.beta(idx_mx_val,idx_wrong)=model.beta(idx_mx_val,idx_wrong)-tau_m*coeff_wrong;
                end
                n_proj2=n_proj2+1;
            else
                n_skip=n_skip+1;
            end
        else %Mistake
            vec=spalloc(1,model.n_cla,2);

            if (delta_true <= model.eta && delta_wrong <= model.eta) || delta_true < eps
                if numel(idx_true)>0
                    model.beta(Yi,idx_true)=model.beta(Yi,idx_true)+coeff_true; % project true
                end
            else
                vec(Yi)=1; % normal update for true
                if numel(model.Kinv{Yi})~=0
                   tmp=[model.Kinv{Yi}, zeros(size(model.Kinv{Yi},1),1);zeros(1,size(model.Kinv{Yi},1)+1)];
                   tmp=tmp+[coeff_true'; -1]*[coeff_true'; -1]'/delta_true;
                else
                   tmp=full(Kii^-1);
                end
                model.Kinv{Yi}=tmp;
                model.Y_cla{Yi}(end+1)=size(model.SV,2)+1;
            end            

            if (delta_true <= model.eta && delta_wrong <= model.eta) || delta_wrong < eps
                if numel(idx_wrong)>0
                    model.beta(idx_mx_val,idx_wrong)=model.beta(idx_mx_val,idx_wrong)-coeff_wrong; % project wrong
                end
            else
                vec(idx_mx_val)=-1; % normal update for wrong
                if numel(model.Kinv{idx_mx_val})~=0
                   tmp=[model.Kinv{idx_mx_val}, zeros(size(model.Kinv{idx_mx_val},1),1);zeros(1,size(model.Kinv{idx_mx_val},1)+1)];
                   tmp=tmp+[coeff_wrong'; -1]*[coeff_wrong'; -1]'/delta_wrong;
                else
                   tmp=full(Kii^-1);
                end
                model.Kinv{idx_mx_val}=tmp;
                model.Y_cla{idx_mx_val}(end+1)=size(model.SV,2)+1;
            end

            if delta_true > model.eta || delta_wrong > model.eta
                model.beta(:,end+1)=vec;
                model.S(end+1)=model.iter;
                model.SV(:,end+1)=X(:,i);
                model.beta2(:,end+1)=0;
            else
                n_proj1=n_proj1+1;
            end
        end
    else
        n_pred=n_pred+1;
    end
    
    model.beta2=model.beta2+model.beta;
    
    model.numSV(model.iter)=numel(model.S);
    
    if mod(i,model.step)==0
        fprintf('#%.0f SV:%5.2f(%d)   pred:%5.2f   skip:%5.2f   proj1:%5.2f   proj2:%5.2f   AER:%5.2f\n', ...
            ceil(i/1000),numel(model.S)/i*100,numel(model.S),n_pred/i*100,n_skip/i*100,n_proj1/i*100,n_proj2/i*100,model.aer(model.iter)*100);
        if isfield(model,'eachRound')~=0
           feval(model.eachRound,model);
        end
    end
end
