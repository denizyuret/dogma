DOGMA - Discriminative Online (Good?) Matlab Algorithms
Online Learning Toolbox for MATLAB
Version 0.31 beta, July 28 2010
Copyright (C) 2009-2011, Francesco Orabona

http://dogma.sourceforge.net



WHAT IS DOGMA?
==============
A "dogma" is "the established belief or doctrine held by a religion,
ideology or any kind of organization: it is authoritative and not to be
disputed, doubted or diverged from" (from Wikipedia). In antithesis with
its name, DOGMA aims to unveil the "mistery" that surrounds online kernel
algorithms, showing how it is easy to design and use them. Are they really
"good" algorithms? Which one is better for a particular applications? How
do they work? You can reply to all these questions just trying them,
playing with them, modifying their codes, with DOGMA!

DOGMA is a MATLAB toolbox for discriminative online learning. It implements
all the state of the art algorithms in a unique framework. The main aim of
the library is simplicity: all the implemented algorithms are easy to be
used, understood, and modified. For this reason, all the implementations
are in plain MATLAB, limiting the use of mex files only when it is strictly
necessary.

The library focuses on linear and kernel online algorithms, mainly
developped in the "relative mistake bound" framework. Examples are
Perceptron, Passive-Aggresive, ALMA, NORMA, SILK, Projectron, RBP,
Banditron, etc.

The toolbox will be constantly updated as soon as new online algorithms are
published in the scientific literature. Submissions from external authors
are also encouraged.



INSTALLATION
============
To install DOGMA simply copy all the files in any directory. The .c and
.cpp files must be compiled using mex. Be sure to configure mex in MATLAB,
using mex -setup. If your are using the 64bit version of MATLAB, you must
compile using the option -largeArrayDims, for example
mex -largeArrayDims chisquare_sparse.c



USAGE
=====
The general usage is the following:

hp.type='rbf';
hp.gamma=0.1;
model_zero=model_init(@compute_kernel,hp);
model=k_pa_train(X,Y,model_zero);

In other words we first define which kernel you want to use, then you
initialize an empty model (model_zero). Finally you pass the empty model to
the specific algorithm you want (for example k_pa_train).

All the algorithms follows the same conventions. In particular:
- the labels must be vectors 1xN, where N is the number of samples;
- all the algorithms whose name starts with 'k' are implemented in dual
  variables, hence they can be used with kernels. To most of them it is
  possible to pass a matrix of training vectors or a precalculated kernel
  matrix. In this latter case the field 'ker' of the model must be empty,
  i.e. [].
- all the algorithms with 'multi' in their name can be used for multiclass
  problems, while all the others can be used only for binary problems.

You can pass the feature vectors or the kernel matrices already calculated:
- feature vectors must be passed as a matrix DxN, where N is the number of
  samples and D is the dimensionality;
- kernel matrices are NxN, where N is the number of samples;
- multi kernel methods accept three dimensional matrices, NxNxF, where N
  is the number of samples and F is the number of kernels.

Most of the online algorithms also calculates a batch solution using an
online-to-batch conversion. Batch solution are indicated by 'beta2' in
algorithms implemented in dual variables and with 'w2' for primal ones.

Run the demo.m to have an example on how to use the algorithms. Other demos
specific to some algorithms are available in the folder "demos".



CODING STYLE
============
In DOGMA we tries our best to follow the coding style in
"The Elements of MATLAB Style" by Richard K. Johnson, Cambridge University Press, 2011.



HOW TO CITE DOGMA?
==================
Please cite the following document: 
 Francesco Orabona, DOGMA: a MATLAB toolbox for Online Learning, 2009. Software available at http://dogma.sourceforge.net
The bibtex format is 
@Manual{Orabona09,
  author =       {Francesco Orabona},
  title =        {{DOGMA}: a {MATLAB} toolbox for Online Learning},
  year =         {2009},
  note =         {Software available at \url{http://dogma.sourceforge.net}}
}

Moreover in source code of every algorithm there is the list of relevant
references that should be cited if you use them in your papers.



TO DO
=====
- Write a better manual...
- Implement all the missing online algorithms!
- Future versions will support Octave (the free alternative to MATLAB) as well.



CONTACT ME
==========
Mail: francesco@orabona.com
Website: http://francesco.orabona.com