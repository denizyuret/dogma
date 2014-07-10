/*   This file is part of the DOGMA library for MATLAB.
     Copyright (C) 2009-2011, Francesco Orabona

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.
 
     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.
 
     You should have received a copy of the GNU General Public License
     along with this program. If not, see <http://www.gnu.org/licenses/>.
 
     Contact the authors: francesco [at] orabona.com 
                          jluo      [at] idiap.ch                        

     This file incorporates work covered by the following copyright and
     permission notice:
 
       Copyright (c) 2009, Peter Vincent Gehler. All rights reserved.

       Redistribution and use in source and binary forms, with or without
       modification, are permitted provided that the following conditions
       are met:
       1. Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
       2. Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in
          the documentation and/or other materials provided with the
          distribution.
       3. The name of the author may not be used to endorse or promote
          products derived from this software without specific prior
          written permission.

       THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
       IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
       WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
       ARE DISCLAIMED.
       IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
       INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
       BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
       LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
       CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
       LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
       ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
       POSSIBILITY OF SUCH DAMAGE.                                       */

#include "mex.h"
#include <math.h>

mxArray *doubleTimesDouble(const mxArray *A1, const mxArray *A2, bool symmetric) {

	unsigned int nDims = mxGetNumberOfDimensions(A1);
	const mwSize *inputdims = mxGetDimensions(A1);
    
	unsigned int n1 = inputdims[0];
	unsigned int n2 = inputdims[1];
	unsigned int m = (nDims==3)?inputdims[2]:1;
	unsigned int nn = n1*n2;
    
    double *K = mxGetPr(A1);
	double *beta = mxGetPr(A2);

	// output arguments
	mxArray* result = mxCreateDoubleMatrix(n1,n2,mxREAL); 
	double *Kbeta = mxGetPr(result); 

	if (symmetric)
	{
		for ( unsigned k=0,knn=0 ; k<m ; k++,knn+=nn )
		{
			if ( beta[k] <= 1e-8)
				continue;

			for ( unsigned int i=0,in = 0 ; i<n1 ; i++,in+=n1 )
				for ( unsigned int j=i ; j<n1 ; j++ )
					Kbeta[in+j] += beta[k] * K[knn+in+j];
		}

		/* symmetrize the output */
		for ( unsigned int i=0,in=0 ; i<n1 ;i++,in+=n1 )
			for ( unsigned int j=i+1 ; j<n1 ; j++)
				Kbeta[j*n1+i] = Kbeta[in+j];

	}
	else
	{
		for ( unsigned k=0,knn=0 ; k<m ; k++,knn+=nn )
		{
			if (beta[k]<=1e-8)
				continue;
			
			for ( unsigned int i=0,in=0 ; i<n2 ; i++,in+=n1 )
				for ( unsigned int j=0 ; j<n1 ; j++ )
					Kbeta[in+j] += beta[k] * K[knn+in+j];
		}
	}
    return result;
}

mxArray *singleTimesDouble(const mxArray *A1, const mxArray *A2, bool symmetric) {

	unsigned int nDims = mxGetNumberOfDimensions(A1);
	const mwSize *inputdims = mxGetDimensions(A1);
    
	unsigned int n1 = inputdims[0];
	unsigned int n2 = inputdims[1];
	unsigned int m = (nDims==3)?inputdims[2]:1;
	unsigned int nn = n1*n2;
    
  	float *K = (float *)mxGetData(A1);
	double *beta = mxGetPr(A2);

	// output arguments
    const mwSize outputdims[2] = {n1, n2};
	mxArray* result = mxCreateNumericArray(2,outputdims,mxSINGLE_CLASS,mxREAL); 
	float *Kbeta = (float *)mxGetData(result); 

	if (symmetric)
	{
		for ( unsigned k=0,knn=0 ; k<m ; k++,knn+=nn )
		{
			if ( beta[k] <= 1e-8)
				continue;

			for ( unsigned int i=0,in = 0 ; i<n1 ; i++,in+=n1 )
				for ( unsigned int j=i ; j<n1 ; j++ )
					Kbeta[in+j] += beta[k] * K[knn+in+j];
		}

		/* symmetrize the output */
		for ( unsigned int i=0,in=0 ; i<n1 ;i++,in+=n1 )
			for ( unsigned int j=i+1 ; j<n1 ; j++)
				Kbeta[j*n1+i] = Kbeta[in+j];

	}
	else
	{
		for ( unsigned k=0,knn=0 ; k<m ; k++,knn+=nn )
		{
			if (beta[k]<=1e-8)
				continue;
			
			for ( unsigned int i=0,in=0 ; i<n2 ; i++,in+=n1 )
				for ( unsigned int j=0 ; j<n1 ; j++ )
					Kbeta[in+j] += beta[k] * K[knn+in+j];
		}
	}
    return result;
}

mxArray *singleTimesSingle(const mxArray *A1, const mxArray *A2, bool symmetric) {

	unsigned int nDims = mxGetNumberOfDimensions(A1);
	const mwSize *inputdims = mxGetDimensions(A1);
    
	unsigned int n1 = inputdims[0];
	unsigned int n2 = inputdims[1];
	unsigned int m = (nDims==3)?inputdims[2]:1;
	unsigned int nn = n1*n2;
    
  	float *K = (float *)mxGetData(A1);
	float *beta = (float *)mxGetData(A2);

	// output arguments
    const mwSize outputdims[2] = {n1, n2};
	mxArray* result = mxCreateNumericArray(2,outputdims,mxSINGLE_CLASS,mxREAL); 
	float *Kbeta = (float *)mxGetData(result); 

	if (symmetric)
	{
		for ( unsigned k=0,knn=0 ; k<m ; k++,knn+=nn )
		{
			if ( beta[k] <= 1e-8)
				continue;

			for ( unsigned int i=0,in = 0 ; i<n1 ; i++,in+=n1 )
				for ( unsigned int j=i ; j<n1 ; j++ )
					Kbeta[in+j] += beta[k] * K[knn+in+j];
		}

		/* symmetrize the output */
		for ( unsigned int i=0,in=0 ; i<n1 ;i++,in+=n1 )
			for ( unsigned int j=i+1 ; j<n1 ; j++)
				Kbeta[j*n1+i] = Kbeta[in+j];

	}
	else
	{
		for ( unsigned k=0,knn=0 ; k<m ; k++,knn+=nn )
		{
			if (beta[k]<=1e-8)
				continue;
			
			for ( unsigned int i=0,in=0 ; i<n2 ; i++,in+=n1 )
				for ( unsigned int j=0 ; j<n1 ; j++ )
					Kbeta[in+j] += beta[k] * K[knn+in+j];
		}
	}
    return result;
}

mxArray *doubleTimesSingle(const mxArray *A1, const mxArray *A2, bool symmetric) {

	unsigned int nDims = mxGetNumberOfDimensions(A1);
	const mwSize *inputdims = mxGetDimensions(A1);
    
	unsigned int n1 = inputdims[0];
	unsigned int n2 = inputdims[1];
	unsigned int m = (nDims==3)?inputdims[2]:1;
	unsigned int nn = n1*n2;
    
  	double *K = mxGetPr(A1);
	float *beta = (float *)mxGetData(A2);

	// output arguments
    const mwSize outputdims[2] = {n1, n2};
	mxArray* result = mxCreateNumericArray(2,outputdims,mxSINGLE_CLASS,mxREAL); 
	float *Kbeta = (float *)mxGetData(result); 

	if (symmetric)
	{
		for ( unsigned k=0,knn=0 ; k<m ; k++,knn+=nn )
		{
			if ( beta[k] <= 1e-8)
				continue;

			for ( unsigned int i=0,in = 0 ; i<n1 ; i++,in+=n1 )
				for ( unsigned int j=i ; j<n1 ; j++ )
					Kbeta[in+j] += beta[k] * K[knn+in+j];
		}

		/* symmetrize the output */
		for ( unsigned int i=0,in=0 ; i<n1 ;i++,in+=n1 )
			for ( unsigned int j=i+1 ; j<n1 ; j++)
				Kbeta[j*n1+i] = Kbeta[in+j];

	}
	else
	{
		for ( unsigned k=0,knn=0 ; k<m ; k++,knn+=nn )
		{
			if (beta[k]<=1e-8)
				continue;
			
			for ( unsigned int i=0,in=0 ; i<n2 ; i++,in+=n1 )
				for ( unsigned int j=0 ; j<n1 ; j++ )
					Kbeta[in+j] += beta[k] * K[knn+in+j];
		}
	}
    return result;
}

/*
 * This function computes
 * A = 0;
 * for i=1:length(beta), A = A + K(:,:,i)*beta(i); end
 * for symmetric and not-symmetric K
 */    
void mexFunction(int , mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	if (nrhs<2 || nrhs>3)
		mexErrMsgTxt("Wrong number of input arguments");

	if (mxGetNumberOfDimensions(prhs[1])!=2)
		mexErrMsgTxt("Second input argument must be 2D");

	unsigned int nDims = mxGetNumberOfDimensions(prhs[0]);
	const mwSize *inputdims = mxGetDimensions(prhs[0]);

	if (nDims<2 || nDims>3)
		mexErrMsgTxt("First input arg must be 2D or 3D");
    
	unsigned int n1 = inputdims[0];
	unsigned int n2 = inputdims[1];
	unsigned int m = (nDims==3)?inputdims[2]:1;
    
    if (mxGetM(prhs[1]) != m)
		mexErrMsgTxt("Input dimensions mismatch");
    
  	bool symmetric = false;
	if (n1==n2)
		if (nrhs>2)
			symmetric = (bool)(*(mxGetPr(prhs[2]))>0);
    
    if ( mxIsSingle(prhs[0]) ) {
        if ( mxIsSingle(prhs[1]) )
            plhs[0] = singleTimesSingle(prhs[0],prhs[1],symmetric);
        else
            plhs[0] = singleTimesDouble(prhs[0],prhs[1],symmetric);
    } else {
        if ( mxIsSingle(prhs[1]) ) {
            plhs[0]=doubleTimesSingle(prhs[0],prhs[1],symmetric);
        } else {
            plhs[0]=doubleTimesDouble(prhs[0],prhs[1],symmetric);
        }
    }
    
}
