function model = k_omcl_multi_train(X,Y,model)
% K_OMCL_MULTI_TRAIN Kernel Online Multi-Cue Learning multiclass algorithm
%
%    MODEL = K_OMCL_MULTI_TRAIN(X,Y,MODEL) trains an classifier
%    according to the Online Multi-Cue Learning algorithm, using kernels.
%
%    MODEL = K_OMCL_MULTI_TRAIN(K,Y,MODEL) trains an classifier
%    according to the Online Multi-Cue Learning algorithm, using kernels.
%    Using pre-computed kernel as input.
%
%    Inputs:
%    K -  N*N*F matrix, K(:,:,i) is a kernel matrix
%    X -  1*F cell, each cell X{f} is a D*N matrix 
%    Y -  Training label, N*1 Vector
%
%    Additional parameters:
%    - model.n_cue is the number of different cues.
%    - model.L1.eta is the sparseness parameter, used to trade-off the
%      performance for sparseness of the classifier for learning each
%      single cue. Note that model.eta is the maximum error on EACH single
%      projection; each projected update has 2 projections.
%      Default value is 0.1.
%    - model.L2.C is the the aggressiveness parameter for the 2nd combining
%      layer, used to trade-off the loss on the current sample with the
%      update on the current hyperplane.
%      Default value is 1.
%
%    Example:
%        See omcl_demo.m
%
%   References:
%     - Jie, L., Orabona, F., & Caputo, B. (2009).
%       An online framework for learning novel concepts over multiple cues.
%       Proceeding of the 9th Asian Conference on Computer Vision
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
%    Contact the authors: francesco [at] orabona.com
%                         jluo      [at] idiap.ch

n = length(Y);   % number of training samples

if isfield(model,'iter')==0
    model.iter=0;
    model.time=[];
    model.beta2=cell(1,model.n_cue);
end

% models 1st layer
for c=1:model.n_cue
    if isfield(model.L1{c}, 'type')==0
        model.L1{c}.type = 'PROJECTRON2';
    end

    if isequal(model.L1{c}.type, 'PA-I')==1 && isfield(model.L1{c}, 'C')==0
        model.L1{c}.C = 1;
    end

    if isequal(model.L1{c}.type, 'Projectron2')==1 && isfield(model.L1{c},'eta')==0
        model.L1{c}.eta=.1;
    end

    if isfield(model.L1{c},'n_cla')==0
        model.L1{c}.n_cla=model.n_cla;
    end

    if isfield(model.L1{c},'iter')==0
        model.L1{c}.iter=0;
        model.L1{c}.beta=[];
        model.L1{c}.beta2=[];
        model.L1{c}.errTot=0;
        model.L1{c}.numSV=zeros(numel(Y),1);
        model.L1{c}.aer=zeros(numel(Y),1);
        model.L1{c}.pred=zeros(numel(Y),1);

        model.L1{c}.n_skip=0;
        model.L1{c}.n_proj1=0;
        model.L1{c}.n_proj2=0;
        model.L1{c}.n_pred=0;

        for i=1:model.L1{c}.n_cla
            model.L1{c}.Kinv{i}=[];
            model.L1{c}.Y_cla{i}=[];
        end
    end
end

% models 2nd layer
if isfield(model.L2,'n_cla')==0
    model.L2.n_cla=max(Y);
end
if isfield(model.L2,'update')==0
    model.L2.update=1; %default update using PA-I
end
if isfield(model.L2,'C')==0
    model.L2.C=1;
end
if isfield(model.L2,'iter')==0
    model.L2.iter=0;
    model.L2.w=zeros(model.n_cla,model.n_cla*model.n_cue);
    model.L2.errTot=0;
    model.L2.numSV=zeros(numel(Y),1);
    model.L2.aer=zeros(numel(Y),1);
    model.L2.pred=zeros(model.L2.n_cla,numel(Y));
end

timerstart = cputime;
h = zeros(model.n_cue*model.n_cla,1);
ht = zeros(model.n_cue*model.n_cla,1);
for i=1:n
    model.iter = model.iter+1;

    Yi=Y(i);

    % Prediction
    pred1 = cell(model.n_cue, 1);
    % 1st layer: Passive-Aggressive/Projectron++
    for c=1:model.n_cue
        pred1{c} = k_layer1_pred(i, c, Yi);
        h((c-1)*model.n_cla+1:c*model.n_cla)=pred1{c}.val_f';
    end
    % 2nd layer: Passive-Aggressive 
    pred2 = layer2_pred(h, Yi);

    % Update
    % 1st layer
    for c=1:model.n_cue
        switch upper(model.L1{c}.type)
            case {'PA-I'}
                pred1{c} = k_pa_update(i,c,Yi,pred1{c});
            case {'PROJECTRON2'}
                pred1{c} = k_projectron2_update(i,c,Yi,pred1{c});
            otherwise
                disp('Unknown base classifier.')
        end 
        ht((c-1)*model.n_cla+1:c*model.n_cla) = pred1{c}.val_f';
    end
    % 2nd layer
    pa_multi_update(ht, Yi);

    if mod(i,model.step)==0 || i==n
        model.time(end+1)=cputime-timerstart;
        fprintf('#%2.0f COMBINATION AER:%5.2f\n', ceil(i/model.step), model.L2.aer(model.iter)*100);
        for c=1:model.n_cue
            fprintf('  Cue %d   SV:%5.2f(%d)   pred:%5.2f   skip:%5.2f   proj1:%5.2f   proj2:%5.2f   AER:%5.2f\n', ...
                c, numel(model.L1{c}.S)/i*100, numel(model.L1{c}.S), model.L1{c}.n_pred/i*100, ...
                model.L1{c}.n_skip/i*100, model.L1{c}.n_proj1/i*100, model.L1{c}.n_proj2/i*100, ...
                model.L1{c}.aer(model.iter)*100);
        end

        if isfield(model,'eachRound')~=0
            if isfield(model, 'outputER')==0
                model.outputER = [];
                model.outputER = feval(model.eachRound, model);
            else
                model.outputER(end+1) = feval(model.eachRound, model);
            end
        end
        timerstart = cputime;
    end
end

% =============== built in functions ===============
% PA-I/Projectron2 predict
function pred = k_layer1_pred(idx_sample, idx_cue, label)
    if numel(model.L1{idx_cue}.S)>0
        if isempty(model.L1{idx_cue}.ker)
            K_f=X(model.L1{idx_cue}.S, idx_sample,idx_cue)';
        else
            K_f=feval(model.L1{c}.ker,model.L1{idx_cue}.SV,X{idx_cue}(:, idx_sample),model.L1{idx_cue}.kerparam);
        end
        val_f=full(model.L1{idx_cue}.beta*K_f);
    else
        K_f=zeros(0,1);
        val_f=zeros(1,model.L1{idx_cue}.n_cla);
    end

    tmp=val_f; tmp(label)=-inf;
    mx_val=max(tmp);
    model.L1{idx_cue}.errTot=model.L1{idx_cue}.errTot+(val_f(label)<=mx_val);
    model.L1{idx_cue}.aer(model.iter)=model.L1{idx_cue}.errTot/model.iter;

    pred.val_f = val_f;
    pred.K_f   = K_f;
end   % end of the function layer1_pred

% PA-I update
% k_pa_update(i,c,Yi,pred1{c})%X{c}(:, i), pred1{c}, Yi);
function pred = k_pa_update(idx_sample, idx_cue, label, pred)
    val_f = pred.val_f;
    K_f   = pred.K_f;

    tmp=val_f; tmp(label)=-inf;
    [mx_val,idx_mx_val]=max(tmp);

    if val_f(label) < mx_val+1 %Margin error or mistake
        model.L1{idx_cue}.S(end+1)=model.iter;

        if isempty(model.L1{idx_cue}.ker)
             Kii = full(X(idx_sample,idx_sample,idx_cue));
             norm_x_square=2*Kii;
        else
             Kii = full(feval(model.L1{idx_cue}.ker,X{idx_cue}(:, idx_sample),X{idx_cue}(:, idx_sample),model.L1{idx_cue}.kerparam));
             norm_x_square=2*Kii;
             model.L1{idx_cue}.SV(:, end+1)=X{idx_cue}(:, idx_sample);
        end

        % PA-I
        new_beta=min((1-(val_f(label)-mx_val))/norm_x_square,model.L1{idx_cue}.C);
        % PA-II
        % new_beta=(1-(val_f(y)-mx_val))/(norm_x_square+1/(2*model.C));

        model.L1{idx_cue}.beta(:,end+1)=spalloc(model.L1{idx_cue}.n_cla,1,2);
        model.L1{idx_cue}.beta(label,end)=new_beta;
        model.L1{idx_cue}.beta(idx_mx_val,end)=-new_beta;

        model.L1{idx_cue}.beta2(:,end+1)=spalloc(model.L1{idx_cue}.n_cla,1,2);

        % if the model is updated, also update the new hyphothesis
        % pred.val_f = full(K_f*model.beta(1:numel(K_f),:));
        % pred.val_f = pred.val_f + full(x(model.iter))*model.beta(end, :);
        pred.val_f = full([ K_f' Kii ]*model.L1{idx_cue}.beta);
    end
    
    model.L1{idx_cue}.beta2=model.L1{idx_cue}.beta2+model.L1{idx_cue}.beta;

    model.L1{idx_cue}.numSV(model.iter)=numel(model.L1{idx_cue}.S);
end   % end of the function pa_update

% projectron2 update
function pred = k_projectron2_update(idx_sample,idx_cue,label,pred)
    val_f = pred.val_f;
    K_f   = pred.K_f;

    idx_true  = [];
    idx_wrong = [];

    tmp=val_f; tmp(label)=-inf;
    [mx_val,idx_mx_val]=max(tmp);

    if val_f(label) < mx_val+1 %Margin error or mistake
        if isempty(model.L1{idx_cue}.ker)
              Kii=full(X(idx_sample,idx_sample,idx_cue));
        else
              Kii=full(feval(model.L1{idx_cue}.ker,X{idx_cue}(:,idx_sample),X{idx_cue}(:,idx_sample),model.L1{idx_cue}.kerparam));
        end
        vec=spalloc(model.L1{idx_cue}.n_cla,1,2);

        delta_true=Kii;
        delta_wrong=Kii;
        if numel(model.L1{idx_cue}.S)>0
            idx_true=model.L1{idx_cue}.Y_cla{label};
            idx_wrong=model.L1{idx_cue}.Y_cla{idx_mx_val};

            if numel(idx_true)>0
                coeff_true=K_f(idx_true)'*model.L1{idx_cue}.Kinv{label};
                % the 'max' is only to prevent numerical problems
                delta_true=max(Kii-coeff_true*K_f(idx_true),0);
            end

            if numel(idx_wrong)>0
                coeff_wrong=K_f(idx_wrong)'*model.L1{idx_cue}.Kinv{idx_mx_val};
                % the 'max' is only to prevent numerical problems
                delta_wrong=max(Kii-coeff_wrong*K_f(idx_wrong),0);
            end
        end

        if val_f(label)>mx_val % Margin error
            loss=1-val_f(label)+mx_val;
            delta=delta_wrong+delta_true;
            if loss-delta/(2*model.L1{idx_cue}.eta)>0       % 2*model.eta because eta is the tollerance on each single projection
                tau_m=min(min(loss/(2*Kii-delta),1),2*(loss-delta/(2*model.L1{idx_cue}.eta))/(2*Kii-delta));
                if numel(idx_true)>0
                    model.L1{idx_cue}.beta(label,idx_true)=model.L1{idx_cue}.beta(label,idx_true)+tau_m*coeff_true;
                end
                if numel(idx_wrong)>0
                    model.L1{idx_cue}.beta(idx_mx_val,idx_wrong)=model.L1{idx_cue}.beta(idx_mx_val,idx_wrong)-tau_m*coeff_wrong;
                end
                model.L1{idx_cue}.n_proj2=model.L1{idx_cue}.n_proj2+1;
            else
                model.L1{idx_cue}.n_skip=model.L1{idx_cue}.n_skip+1;
            end
        else % Mistake
            if (delta_true <= model.L1{idx_cue}.eta && delta_wrong <= model.L1{idx_cue}.eta) || delta_true < eps
                if numel(idx_true)>0
                    % project true
                    model.L1{idx_cue}.beta(label,idx_true)=model.L1{idx_cue}.beta(label,idx_true)+coeff_true;
                end
            else
                vec(label)=1; % normal update for true
                if numel(model.L1{idx_cue}.Kinv{label})~=0
                   tmp=[model.L1{idx_cue}.Kinv{label}, zeros(size(model.L1{idx_cue}.Kinv{label},1),1);zeros(1,size(model.L1{idx_cue}.Kinv{label},1)+1)];
                   tmp=tmp+[coeff_true'; -1]*[coeff_true'; -1]'/delta_true;
                else
                   tmp=full(Kii^-1);
                end
                model.L1{idx_cue}.Kinv{label}=tmp;
                model.L1{idx_cue}.Y_cla{label}(end+1)=numel(model.L1{idx_cue}.S)+1;
            end

            if (delta_true <= model.L1{idx_cue}.eta && delta_wrong <= model.L1{idx_cue}.eta) || delta_wrong < eps
                if numel(idx_wrong)>0
                    % project wrong
                    model.L1{idx_cue}.beta(idx_mx_val,idx_wrong)=model.L1{idx_cue}.beta(idx_mx_val,idx_wrong)-coeff_wrong;
                end
            else
                vec(idx_mx_val)=-1; % normal update for wrong
                if numel(model.L1{idx_cue}.Kinv{idx_mx_val})~=0
                   tmp=[model.L1{idx_cue}.Kinv{idx_mx_val}, zeros(size(model.L1{idx_cue}.Kinv{idx_mx_val},1),1); ...
                        zeros(1,size(model.L1{idx_cue}.Kinv{idx_mx_val},1)+1)];
                   tmp=tmp+[coeff_wrong'; -1]*[coeff_wrong'; -1]'/delta_wrong;
                else
                   tmp=full(Kii^-1);
                end
                model.L1{idx_cue}.Kinv{idx_mx_val}=tmp;
                model.L1{idx_cue}.Y_cla{idx_mx_val}(end+1)=numel(model.L1{idx_cue}.S)+1;
            end

            if delta_true > model.L1{idx_cue}.eta || delta_wrong > model.L1{idx_cue}.eta
                model.L1{idx_cue}.beta(:,end+1)=vec;
                model.L1{idx_cue}.S(end+1)=model.iter;
                if ~isempty(model.L1{idx_cue}.ker)
                    model.L1{idx_cue}.SV(:,end+1)=X{idx_cue}(:,idx_sample);
                end
                model.L1{idx_cue}.beta2(:,end+1)=0;
            else
                model.L1{idx_cue}.n_proj1=model.L1{idx_cue}.n_proj1+1;
            end
        end

        % if the model is updated, also update the new hyphothesis
        pred.val_f = full(model.L1{idx_cue}.beta(:,1:numel(K_f))*K_f);
        pred.val_f = pred.val_f + Kii*vec;
    else
       model.L1{idx_cue}.n_pred=model.L1{idx_cue}.n_pred+1;
    end

    model.L1{idx_cue}.beta2=model.L1{idx_cue}.beta2+model.L1{idx_cue}.beta;

    model.L1{idx_cue}.numSV(model.iter)=numel(model.L1{idx_cue}.S);
end   % end of the function projectron2_update

% pa predict
function pred = layer2_pred(input, label)
    val_f=model.L2.w*input;
    tmp=val_f; tmp(label)=-inf;
    mx_val=max(tmp);
    model.L2.errTot=model.L2.errTot+(val_f(label)<=mx_val);
    model.L2.aer(model.iter)=model.L2.errTot/model.iter;
    model.pred(:,model.iter)=val_f;
    pred = val_f;
end   % end of pa predict

% pa update
function pa_multi_update(ht, y)
    % receive the updated hypothesis from 1st layer
    val_f=model.L2.w*ht;

    tmp=val_f; tmp(y)=-inf;
    [mx_val,idx_mx_val]=max(tmp);

    if val_f(y)<1+mx_val
        norm_x_square=2*norm(ht)^2;
        if model.L2.update==1
            new_beta=min((1-(val_f(y)-mx_val))/norm_x_square,model.L2.C);
        else
            new_beta=(1-(val_f(y)-mx_val))/(norm_x_square+1/(2*model.L2.C));
        end
        model.L2.w(y,:)=model.L2.w(y,:)+new_beta*ht';
        model.L2.w(idx_mx_val,:)=model.L2.w(idx_mx_val,:)-new_beta*ht';
        model.L2.S(end+1)=model.L2.iter;
    end
    model.L2.numSV(model.iter)=numel(model.L2.S);

    % update final w (online to batch conversion)
    alpha = model.L2.w;
    for cc=1:model.n_cue
        beta  = full(model.L1{cc}.beta);
        model.beta{cc} = alpha(:, (cc-1)*model.n_cla+1:cc*model.n_cla)*beta;
    end
    for cc=1:numel(model.beta2)
        if size(model.beta2{cc}, 2)<size(model.beta{cc}, 2)
            model.beta2{cc}(:, end+1)=0;
        end
        model.beta2{cc} = model.beta2{cc}+model.beta{cc};
    end
end   % end of pa update

end