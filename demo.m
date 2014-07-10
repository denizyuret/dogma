% DEMO for budget algorithms
% The following algorithms are used: 
% - k_forgetron_st
% - k_oisvm_train
% - k_pa_train
% - k_projectron_train
% - k_projectron2_train
% - k_perceptron_train

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

clc
clear
close all

% create random 2-dimensional problem
n = 20000;
x = randn(2,n);
% random hyperplane
w = randn(2,1);
% create labels
y = sign(w'*x);
% add noise on the labels flipping 10% of the labels
rp = randperm(n);
y(rp(1:n*0.1)) = -y(rp(1:n*0.1));

% first half of the samples for training
x_tr = x(:,1:n/2);
y_tr = y(1:n/2);
% second half of the samples for testing
x_te = x(:,n/2+1:end);
y_te = y(n/2+1:end);

% select kernel
hp.type = 'rbf'; %g aussian kernel: exp(-gamma |x_i-x_j|^2)
hp.gamma = 1;

% inizialize an empty model
model_bak = model_init(@compute_kernel,hp);


% choose the parameter 'C' of the PA-I
model_bak.C = 1;
% train PA-I
fprintf('Training PA-I model...\n');
model_pa1 = k_pa_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',numel(model_pa1.beta));
fprintf('Number of support vectors averaged solution:%d\n',numel(model_pa1.beta2));
fprintf('Testing last solution...');
pred_pa1_last = model_predict(x_te,model_pa1,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_pa1_last~=y_te))/numel(y_te)*100);
fprintf('Testing averaged solution...');
pred_pa1_av = model_predict(x_te,model_pa1,1);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_pa1_av~=y_te))/numel(y_te)*100);


% train Perceptron
fprintf('Training Perceptron model...\n');
model_perceptron = k_perceptron_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',numel(model_perceptron.beta));
fprintf('Number of support vectors averaged solution:%d\n',numel(model_perceptron.beta2));
fprintf('Testing last solution...');
pred_perceptron_last = model_predict(x_te,model_perceptron,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_perceptron_last~=y_te))/numel(y_te)*100);
fprintf('Testing averaged solution...');
pred_perceptron_av = model_predict(x_te,model_perceptron,1);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_perceptron_av~=y_te))/numel(y_te)*100);


% set the parameter 'eta' of the Projectron
model_bak.eta = 0.01;
% train Projectron
fprintf('Training Projectron model...\n');
model_projectron = k_projectron_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',numel(model_projectron.beta));
fprintf('Number of support vectors averaged solution:%d\n',numel(model_projectron.beta2));
fprintf('Testing last solution...');
pred_projectron_last = model_predict(x_te,model_projectron,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_projectron_last~=y_te))/numel(y_te)*100);
fprintf('Testing averaged solution...');
pred_projectron_av = model_predict(x_te,model_projectron,1);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_projectron_av~=y_te))/numel(y_te)*100);


% set the parameter 'eta' of the Projectron++
model_bak.eta = 0.01;
% train Projectron++
fprintf('Training Projectron++ model...\n');
model_projectron2 = k_projectron2_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',numel(model_projectron2.beta));
fprintf('Number of support vectors averaged solution:%d\n',numel(model_projectron2.beta2));
fprintf('Testing last solution...');
pred_projectron2_last = model_predict(x_te,model_projectron2,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_projectron2_last~=y_te))/numel(y_te)*100);
fprintf('Testing averaged solution...');
pred_projectron2_av = model_predict(x_te,model_projectron2,1);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_projectron2_av~=y_te))/numel(y_te)*100);


% set maximum number of support vectors for RBP
model_bak.maxSV = numel(model_projectron.beta);
% train RBP
fprintf('Training RBP...\n');
model_rbp = k_perceptron_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',numel(model_rbp.beta));
fprintf('Testing last solution...');
pred_rbp_last = model_predict(x_te,model_rbp,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_rbp_last~=y_te))/numel(y_te)*100);


% set maximum number of support vectors for Forgetron
model_bak.maxSV = numel(model_projectron.beta);
% train Forgetron
fprintf('Training Forgetron (self-tuned)...\n');
model_forgetron = k_forgetron_st_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',numel(model_forgetron.beta));
fprintf('Testing last solution...');
pred_forgetron_last = model_predict(x_te,model_forgetron,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_forgetron_last~=y_te))/numel(y_te)*100);


% set the parameter 'eta' of the OISVM
model_bak.eta = 0.01;
% set the parameter 'C' of the OISVM
model_bak.C = 1;
% train OISVM
fprintf('Training OISVM...\n');
model_oisvm = k_oisvm_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',numel(model_oisvm.beta));
fprintf('Testing last solution...');
pred_oisvm_last = model_predict(x_te,model_oisvm,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_oisvm_last~=y_te))/numel(y_te)*100);


% plot error curves
figure(1)
plot(model_pa1.aer(100:end),'k')
hold on
plot(model_perceptron.aer(100:end),'c')
plot(model_projectron.aer(100:end),'b')
plot(model_projectron2.aer(100:end),'m')
plot(model_rbp.aer(100:end),'r')
plot(model_forgetron.aer(100:end),'y')
plot(model_oisvm.aer(100:end),'g')
grid
legend('PA-I','Perceptron','Projectron','Projectron++','RBP','Forgetron','OISVM')
xlabel('Number of Samples')
ylabel('Average Online Error')

% plot support vector curves
figure(2)
plot(model_pa1.numSV,'k')
hold on
plot(model_perceptron.numSV,'c')
plot(model_projectron.numSV,'b')
plot(model_projectron2.numSV,'m')
plot(model_rbp.numSV,'r')
plot(model_forgetron.numSV,'y')
plot(model_oisvm.numSV,'g')
legend('PA-I','Perceptron','Projectron','Projectron++','RBP','Forgetron','OISVM','Location','NorthWest')
grid
xlabel('Number of Samples')
ylabel('Number of Support Vectors')
