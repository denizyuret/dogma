function model = k_oisvm_train(X,Y,model)
% K_OISVM_TRAIN Kernel Online Independent SVM algorithm
%
%    MODEL = K_OISVM_TRAIN(X,Y,MODEL) trains an classifier according to the
%    Online Independent SVM algorithm, using kernels.
%
%    Additional parameters:
%    - model.C is the weight of the error, used to reduce the amount of
%      regularization.
%      Default value is 1.
%    - model.eta is the sparseness parameter, used to trade-off the
%      performance for sparseness of the classifier.
%      Default value is 0.1.
%
%   References:
%     - Orabona, F., Castellini, C., Caputo, B., Jie, L., & Sandini, G. (2010).
%       Online Independent Support Vector Machines.
%       Pattern Recognition 43(4), (pp. 1402-1412).

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

    model.num_ker_eval=0;
    model.X=[];
    model.Y=[];
    model.K=zeros(100,100);
    model.x=0;
    model.sv=[];
    model.ss=0;
    model.KbInv=[];
end

if isfield(model,'C')==0
    model.C=1;
end

if isfield(model,'eta')==0
    model.eta=.1;
end

C=model.C;
ker=model.ker;
kerparam=model.kerparam;

if ~isfield(model,'maxiter'),  model.maxiter  = 20;  end
if ~isfield(model,'rows_mem'), model.rows_mem = 100; end
if ~isfield(model,'cols_mem'), model.cols_mem = 100; end
  
maxRow=size(model.K,1);
maxCol=size(model.K,2);

for curr=1:n
        
    model.iter=model.iter+1;
    dimS=length(model.S);
    
    model.X=[model.X , X(:,curr)];
    model.Y=[model.Y , Y(curr)];
    ultimo=size(model.X,2);
    
    last_col=[Y(curr); feval(ker,model.X,model.S,ultimo,kerparam) * Y(curr)];
    model.num_ker_eval=model.num_ker_eval+numel(model.S);
    model.K(1:dimS+1,ultimo)=last_col;
    if ultimo==maxCol
        tmp=zeros([size(model.K,1) size(model.K,2)+model.cols_mem]);
        tmp(1:dimS+1,1:ultimo)=model.K(1:dimS+1,1:ultimo);
        %tmp(:,1:curr)=K(:,1:curr); %is it faster copying even the zeros?
        model.K=tmp;
        maxCol=size(model.K,2);
    end
    
    if dimS>0
        colonna2=model.K(2:dimS+1,ultimo)*Y(curr);
        amin=model.KbInv*colonna2;
        delta=feval(ker,X,curr,curr,kerparam)-colonna2'*amin;
        model.num_ker_eval=model.num_ker_eval+1;
        
        if (delta>model.eta)
            model.KbInv=[model.KbInv, zeros(dimS,1);zeros(1,dimS+1)];
            model.KbInv=model.KbInv+[amin; -1]*[amin; -1]'/delta;

            model.S = [model.S ultimo];
            riga=feval(ker,model.X,ultimo,1:ultimo,kerparam).*model.Y(1:ultimo);
            model.num_ker_eval=model.num_ker_eval+ultimo;
            model.K(dimS+2,1:ultimo)=riga;
            
            % update Cholesky decomposition of the hessian
            d  = length(model.S);
            h = [0; model.K(d+1,model.S)'.*model.Y(model.S)'] + C * model.K(1:d+1,model.sv) * model.K(d+1,model.sv)';
            h2 = h(end,:);
            h2 = h2 + 1e-10*h2*eye(size(h,2)); % Ridge is only for numerical reason
            h3 = model.hess' \ h(1:d,:);
            h4 = sqrt(h2-h3'*h3);
            model.hess = [[model.hess h3]; [zeros(1,d) h4]];
            
            model.ss=[model.ss;sum(riga(model.sv))];
            if dimS+2==maxRow
                tmp=zeros([size(model.K,1)+model.rows_mem size(model.K,2)]);
                tmp(1:size(model.K,1),1:ultimo)=model.K(1:size(model.K,1),1:ultimo);
                model.K=tmp;
                maxRow=size(model.K,1);
            end
        end
    else
        model.S = ultimo;
        riga=feval(ker,model.X,ultimo,1:ultimo,kerparam).*model.Y(1:ultimo)';
        model.num_ker_eval=model.num_ker_eval+ultimo;
        model.K(2,1:ultimo)=riga;
        % update Cholesky decomposition of the hessian
        model.hess=[1e-5,0;0,sqrt(model.K(2,1)*Y(1))];
        model.ss=[model.ss;sum(riga(model.sv))];
        model.KbInv=(model.K(2,1)*Y(1))^-1;
    end
            
    outNew = model.K(1:length(model.x),ultimo)'*model.x;
    
    model.errTot=model.errTot+(sign(outNew)<=0);
    model.aer(model.iter)=model.errTot/model.iter;

    model.pred(model.iter)=outNew*Y(curr);
    
    if outNew<1
      K2 = model.K(1:size(model.hess,1),1:ultimo);
      
      model.hess = cholupdate(model.hess,sqrt(C)*K2(:,ultimo),'+');
      model.ss = model.ss+K2(:,ultimo);
      
      iter = 0;

      model.sv = [model.sv,ultimo];
      sv_bool = zeros(1,ultimo);
      sv_bool(model.sv) = 1;

      while iter < model.maxiter
        iter = iter + 1;
        % Take a few Newton step (no line search). By writing out the
        % equations, this simplifies to following equation:
        model.x = C*(model.hess \ (model.hess' \ model.ss));
        
        out = model.x'*K2;      % Recompute the outputs...

        new_sv_bool = (out<1);
        new_sv = find(new_sv_bool);

        % The set of errors has changed (and so the Hessian), so we update
        % the Cholesky decomposition of the Hessian.
        change=0;
        for i=find(new_sv_bool>sv_bool)
          model.hess = cholupdate(model.hess,sqrt(C)*K2(:,i),'+');
          model.ss = model.ss+K2(:,i);
          change=1;
        end
        for i=find(sv_bool>new_sv_bool)
          model.hess = cholupdate(model.hess,sqrt(C)*K2(:,i),'-');
          model.ss = model.ss-K2(:,i);
          change=1;
        end

        % Compute the objective function (it is not needed by the algorithm)
        % obj = 0.5* (norm(hess*x)^2 - 2*C*sum(out(new_sv)) + C*length(new_sv));
        % fprintf(['\rNb basis = %d (%d), iter Newton = %d, Obj = %.2f, ' ...
        %     'Nb errors = %d   '],length(hess)-1,length(find(x))-1,iter,obj,length(sv));
        
        if change==0
            break;
        end
        model.sv = new_sv;
        sv_bool=new_sv_bool;
      end
    end
    
    model.numSV(model.iter)=numel(model.S);
    
    if (mod(curr,model.step)==0)
        fprintf('#%.0f SV:%5.2f(%d)\tAER:%5.2f\n', ...
        ceil(curr/1000),numel(model.S)/curr*100,numel(model.S),model.aer(model.iter)*100);
    end
end

model.beta = [model.x(2:end); zeros(numel(model.S)-(numel(model.x)-1),1)]';
model.b = model.x(1);
