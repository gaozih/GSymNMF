/**
author: Ziheng Gao
**/
#include "math.h"
#include "mex.h" 
#include <time.h>
#include <stdlib.h>

void ggcHiter(double *GH,double *HH,/*double* L,*/ double *H,/*double lamda,*/double miu, double tol, int n, int k, int maxinner)
{
	// initial maximum function value decreasing over all coordinates. 
	double init=0; 

	// Diagonal of HH
	double *HH_d = (double *)malloc(sizeof(double)*k);
	for ( int i=0 ; i<k ; i++ )
		HH_d[i] = HH[i*k+i];
    
    //double *L_d = (double*)malloc(sizeof(double)*n);
    //for (int i = 0; i < n; i++)
    //    L_d[i] = L[i*n+i];
    
	// Create SHt : store step size for each variables 
	double *SHt = (double *)malloc(sizeof(double)*k);

	// Get init value 
	for ( int i=0, nowidx=0 ; i<n ; i++ )
	{
		for ( int j=0 ; j<k ; j++, nowidx++ )
		{
			double s = GH[nowidx]/(HH_d[j]/*+lamda*L_d[i]*/+miu);
			s = H[nowidx]-s;
			if ( s< 0)
				s=0;
			s = s-H[nowidx];
			double diffobj = (-1)*s*GH[nowidx]-0.5*(HH_d[j]/*+lamda*L_d[i]*/+miu)*s*s;
			if ( diffobj > init )
				init = diffobj;
		}
	}

	// stopping condition

	// coordinate descent 
	for ( int p=0 ; p<n ; p++)
	{
		double *GHp = &(GH[p*k]);
		double *Hp = &(H[p*k]);
		for ( int winner = 0 ; winner < maxinner ; winner++)
		{
			// find the best coordinate 
			int q = -1;
			double bestvalue = 0;

			for ( int i=0; i<k ; i++ )
			{
				double ss = GHp[i]/(HH_d[i]/*+lamda*L_d[i]*/+miu);
				ss = Hp[i]-ss;
				if (ss < 0)
					ss=0;
				ss = ss-Hp[i];
				SHt[i] = ss;
				double diffobj = (-1)*(ss*GHp[i]+0.5*(HH_d[i]/*+L_d[i]*lamda*/+miu)*ss*ss);
				if ( diffobj > bestvalue ) 
				{
					bestvalue = diffobj;
					q = i;
				}
			}
			if ( q==-1 )
				break;

			Hp[q] += SHt[q];
			int base = q*k;
			for ( int i=0 ; i<k ; i++ )
				GHp[i] += SHt[q]*HH[base+i];
            //for(int i = 0; i < n; i++)
            //    GH[i*k+q] += lamda*SHt[q]*L[i*n+i];
            
            GHp[q] += miu*SHt[q];
			if ( bestvalue < init*tol)
				break;
		}
	}
	free(HH_d);
	free(SHt);
}


void usage()
{
	printf("Error calling doiter.\n");
	printf("Usage: Wnew = ggc_h(GW, HH^T, W, tol, maxinner)\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *GW, *HH, *W, *values;
	double tol,miu;
	int maxinner, k, n;

	// Input Arguments

	GW = mxGetPr(prhs[0]);
	k = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);

	HH = mxGetPr(prhs[1]);
	if ( (mxGetM(prhs[1]) != k) || (mxGetN(prhs[1])!=k) ) {
		usage();
		printf("Error: %d %d HH^T should be a %d by %d matrix\n", mxGetM(prhs[2]), mxGetN(prhs[2]), k, k);
	}
    //L = mxGetPr(prhs[2]);
	W = mxGetPr(prhs[2]);
	if ( (mxGetM(prhs[2])!=k) || (mxGetN(prhs[2])!=n) ) {
		usage();
		printf("Error: W should be a %d by %d matrix", k, n);
	}

	//values = mxGetPr(prhs[3]);
    //lamda = values[0];
	values = mxGetPr(prhs[3]);
	miu = values[0];

	values = mxGetPr(prhs[4]);
	tol = values[0];

	values = mxGetPr(prhs[5]);
	maxinner = values[0];

	/// Output arguments
	plhs[0] = mxCreateDoubleMatrix(k,n,mxREAL);
	double *Wout = mxGetPr(plhs[0]);
		
	ggcHiter(GW,HH,W,miu,tol, n, k, maxinner);

	for ( int i=0 ; i<n*k ; i++ )
		Wout[i] = W[i];
}
