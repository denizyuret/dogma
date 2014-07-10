function err = precrec(margins, true_labels)
% PRECREC Precision and Recall calculation
%
%    ERR = PRECREC(MARGINS, TRUE_LABELS) calculates the
%    Precision and Recall, F1, Accuracy and AUC.
%
%    Additional parameters:
%      none
%
%   References:
%    Hand,D. J., & Till, R. J. (2001)
%    A simple generalisation of the area under the ROC curve for multiple
%    class classification problems.
%    Machine Learning, 45, (pp. 171-186).

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

margins=margins(:);
pred_labels=sign(margins);
true_labels=true_labels(:);

tp=numel(find(pred_labels==1 & true_labels==1));
fp=numel(find(pred_labels==1 & true_labels==-1));
fn=numel(find(pred_labels==-1 & true_labels==1));
err.prec=tp/(tp+fp+eps);
err.rec=tp/(tp+fn+eps);
err.f1=2*err.prec*err.rec/(err.prec+err.rec+eps);
err.acc=numel(find(pred_labels==true_labels))/numel(pred_labels);

nTarget     = numel(find(true_labels == 1));
nBackground = numel(find(true_labels == -1));
[sorted_pred,idx_sort]=sort(margins);
%for i=1:numel(pred_labels)
%    idx_sort(i)=mean(find(margins(i)==sorted_pred));
%end
idx_sort=tiedrank(margins);
err.auc=(sum(idx_sort(true_labels == 1)) - nTarget*(nTarget+1)/2 ) / (nTarget * nBackground);

% for i=1:numel(sorted_pred)
%     pred_labels=sign(margins+sorted_pred(i));
%     tp=numel(find(pred_labels==1 & true_labels==1));
%     fp=numel(find(pred_labels==1 & true_labels==-1));
%     tpr(i)=tp/numel(find(true_labels==1));
%     fpr(i)=fp/numel(find(true_labels==-1));
%     %fn=numel(find(pred_labels==-1 & true_labels==1));
%     %prec(i)=tp/(tp+fp+eps);
%     %rec(i)=tp/(tp+fn+eps);
% end
% %plot(rec,prec)
% plot(fpr,tpr)
% 
