% DOGMA - Discriminative Online Good Matlab Algorithms
% Version 0.31 beta, July 28 2011
% Copyright (C) 2009-2011, Francesco Orabona.
%
% Please refer to your MATLAB documentation on how to add DOGMA to your
% MATLAB search path.
%
% Online learning algorithms.
%   adagrad_rda_sql2_diag_train - Adagrad with RDA updates, squared L2 regularizer, hinge loss, and diagonal matrix.
%   arow_train                  - Adaptive Regularization Of Weight Vectors algorithm
%   arow_diag_train             - Adaptive Regularization Of Weight Vectors algorithm, diagonal version
%   banditron_multi_train       - Banditron
%   k_alma2_train               - Kernel Approximate Maximal Margin Algorithm, with the 2-norm
%   k_forgetron_st_train        - Kernel Forgetron, 'self-tuned' variant
%   k_oisvm_train               - Kernel Online Independent SVM
%   k_om2_multi_train           - Kernel Online Multi-class Multi-kernel Learning
%   k_om2_mp_multi_train        - Kernel Online Multi-class Multi-kernel Learning, multiple passes
%   k_omcl_multi_train          - Kernel Online Multi Cue Learning multiclass
%   k_pa_train                  - Kernel Passive-Aggressive, PA-I and PA-II variants
%   k_pa_multi_train            - Kernel Passive-Aggressive multiclass, PA-I and PA-II variants
%   k_perceptron_train          - Kernel Perceptron/Random Budget Perceptron
%   k_perceptron_multi_train    - Kernel Perceptron/Random Budget Perceptron multiclass
%   k_projectron_train          - Kernel Projectron
%   k_projectron2_train         - Kernel Projectron++
%   k_projectron2_multi_train   - Kernel Projectron++ multiclass
%   k_sop_train                 - Kernel Second-order Perceptron
%   mms_multi_train             - Max Margin Set Learning algorithm
%   narow_train                 - Narrow Adaptive Regularization Of Weight Vectors algorithm
%   pa_train                    - Passive-Aggressive, PA-I and PA-II variants
%   pa_multi_train              - Passive-Aggressive multiclass, PA-I and PA-II variants
%   perceptron_train            - Perceptron
%   pnorm_train                 - p-Norm
%   sop_train                   - Second-order Perceptron
%   sop_adapt_train             - Second-order Perceptron, adaptive version
%   vaw_train                   - Vovk–Azoury–Warmuth forecaster 
%
% Online optimization algorithms.
%   k_pegasos_train             - Kernel Pegasos
%   k_obscure_train             - Online-Batch Strongly Convex mUlti kErnel leaRning
%   k_obscure_online_train      - Online-Batch Strongly Convex mUlti kErnel leaRning - 1st phase
%   k_obscure_batch_train       - Online-Batch Strongly Convex mUlti kErnel leaRning - 2nd phase
%   k_ufomkl_multi_train        - Ultra Fast Optimization for multiclass Multi Kernel Learning
%   k_ufomkl_logistic_train     - Ultra Fast Optimization for Multi Kernel Learning, with logistic loss
%   k_ufomkl_train              - Ultra Fast Optimization for Multi Kernel Learning
%
% Selective sampling algorithms.
%   bbq_train                   - Bound on Bias Query Algorithm
%   dgs_mod_train               - Modified Dekel-Gentile-Sridharan selective sampler algorithm
%   k_dgs_mod_train             - Kernel Modified Dekel-Gentile-Sridharan selective sampler algorithm
%   k_sel_perc_train            - Kernel Selective Perceptron
%   k_sel_ada_perc_train        - Kernel Selective Perceptron with Adaptive Sampling
%   k_sole_train                - Kernel Second Order Label Efficient
%   k_ss_train                  - Kernel Selective Sampler
%   k_ssmd_train                - Kernel Selective Sampling Mistake Driven
%   sel_perc_train              - Selective Perceptron
%   sel_ada_perc_train          - Selective Perceptron with Adaptive Sampling
%   sole_train                  - Second Order Label Efficient
%   ss_train                    - Selective Sampler
%   ssmd_train                  - Selective Sampling Mistake Driven
%
% Auxiliary functions.
%   model_init                  - General inizializiation function
%   model_predict               - General prediction function
%   model_mc_init               - Inizializiation function for Multi Cue Learning
%
% Miscellaneous.
%   compute_kernel              - Calculate the kernel values
%   demo                        - Demo of many classification algorithms
%   precrec                     - Precision and Recall calculation
%   randnorm                    - Sample from multivariate normal
%   shuffledata                 - Shuffle input and output data