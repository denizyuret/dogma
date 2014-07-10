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
     along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
     Contact the author: francesco [at] orabona.com */

#include "mex.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void mexFunction( int nlhs,       mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] 
                 )
                
{   const double *h1, *h2;
    mwSize nrow1, nrow2;
    mwIndex *ir1, *ir2, i1, i2;
    mwIndex maxc1, maxc2;
    double ret=0;
    
    /* Check I/O number */
    if (nlhs!=1) {
        mexErrMsgTxt("Incorrect number of outputs");
    }
    if (nrhs!=2) {
        mexErrMsgTxt("Incorrect number of inputs");
    }
    
    /* Brief error checking */
    
    /*ncol1 = mxGetN(prhs[0]);
    ncol2 = mxGetN(prhs[1]);*/
    nrow1 = mxGetM(prhs[0]);
    nrow2 = mxGetM(prhs[1]);
    
    /*if ((ncol!=mxGetM(prhs[1])) || (mxGetN(prhs[1])!=1)) {
        mexErrMsgTxt("Wrong input dimensions");
    }
    if (!mxIsSparse(prhs[0])) {
        mexErrMsgTxt("Matrix must be sparse");
    }*/
    
    
    /* Allocate output */
    /*plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[0]), 1, mxREAL);*/

    
    /* I/O pointers */
    ir1 = mxGetIr(prhs[0]);      /* Row indexing      */
    ir2 = mxGetIr(prhs[1]);      /* Row indexing      */    
    h1  = mxGetPr(prhs[0]);      /* Non-zero elements */
    h2  = mxGetPr(prhs[1]);      /* Rhs vector        */
    maxc1 = mxGetJc(prhs[0])[1];
    maxc2 = mxGetJc(prhs[1])[1];

    
    i1=0;
    i2=0;

    while(i1<maxc1 && i2<maxc2) {            /* Loop through columns */
        if (ir1[i1]==ir2[i2]) {
            ret=ret+MIN(h1[i1],h2[i2]);
            i1++;
            i2++;
        } else {
            if (ir1[i1]>ir2[i2])
                i2++;
            else
                i1++;
        }
    }

    plhs[0] = mxCreateDoubleScalar(ret);
}
