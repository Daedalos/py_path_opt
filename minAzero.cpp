// Working Version to include Time Delays --URI

//#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "optimization.h"
#include <iostream>
#include <string>
#include <fstream>
using namespace std;
using namespace alglib;


#define DT 0.01     // time step size

// These were previously defined in func.cpp. Moved here to keep
// constant in one place --URI 
#define NX 5   	     // dim of state variable + number of parameters 
#define ND 5         // dim of state variable

#define NT 100       // number of time steps
#define NMEA 1       // number of measurements

#define NPATH 1    // number of paths
#define NBETA 30     // maximal beta
const int BETASTART = 0; // possible ot start at Beta!=0 --URI

real_2d_array Ydata;
const bool generate_paths = false;

const int NTD = 1;

const int taus[NTD] = {20};

int measIdx[NMEA];

void simple_mmult(real_2d_array &A, real_2d_array &B, real_2d_array &C);
#include "func_noparam.cpp"  // func.cpp uses DT, so include after defining#include "func_noparam.h"  // func.cpp uses DT, so include after defining

void readdata(real_2d_array &data){
	FILE *fp;
	fp = fopen("./dataN_D5_dt0.01_noP.txt","r");
	int i,j;
	for(i=0;i<NT;i++)
		for(j=0;j<NX;j++)
			fscanf(fp,"%lf", &data[i][j]);
}

// slices i^th row of matrix, into output
void slice(real_2d_array &matrix, int i, real_1d_array &output){
	int j;
	int c = matrix.cols();
	for(j=0;j<c;j++)
		output[j] = matrix[i][j];
}

// Opposite of slice --URI	
void insert(real_2d_array &matrix, int i, real_1d_array &output){
  int j;
  int c = matrix.cols();
  for(j=0; j<c; j++)
    matrix[i][j] = output[j];

}

// wrapper for alglib::rmatrixgemm function to matrix multiply
// eliminate extra arguments

void simple_mmult(real_2d_array &A, real_2d_array &B, real_2d_array &C){
  rmatrixgemm(NX,NX,NX,1, A,0,0,0, B,0,0,0, 0, C,0,0);
}


void action_grad(const real_1d_array &x, double &action, real_1d_array &grad, void *ptr) {
	real_2d_array XX, grad_m, test;
   	XX.setlength(NT,NX);
	grad_m.setlength(NT,NX);
	test.setlength(NT,NX);
	int i,j,k;
	for(i=0;i<NT;i++){
		for(j=0;j<NX;j++){
			XX[i][j] = x[NX*i+j];
		}
	}
	//reshapevector2matrix(x,XX);
	int beta = *(int*)ptr;
	double Rm=2, Rf=0.005*pow(2,beta),tmpd;
	real_1d_array x1,x2,f1,f2,tmpa1,tmpa2;
	x1.setlength(NX);
	x2.setlength(NX);
	f1.setlength(NX);
	f2.setlength(NX);
	tmpa1.setlength(NX);
	tmpa2.setlength(NX);
	real_2d_array J1, J2, tmpm1, tmpm2;
	J1.setlength(NX,NX);
	J2.setlength(NX,NX);
	tmpm1.setlength(NX,NX);
	tmpm2.setlength(NX,NX);

	action = 0;
	for(i=0;i<NT;i++)
		for(j=0;j<NX;j++)
			grad_m[i][j]=0;
	for(i=0;i<NMEA;i++){
		for(j=0;j<NT;j++){
			
			tmpd = (XX[j][2*i]-Ydata[j][2*i]);
			
			action = action + Rm*tmpd*tmpd;
			grad_m[j][2*i] = grad_m[j][2*i] + 2*Rm*tmpd;
		}
	}
	
	for(j=0;j<(NT-1);j++){
		slice(XX, j, x1);
		slice(XX, j+1, x2);
		func_origin(x1, f1);
		func_origin(x2, f2);
		
		func_DF(x1,J1);
		func_DF(x2,J2);
		for(i=0;i<NX;i++){
			for(k=0;k<NX;k++){
				tmpm1[i][k]=-0.5*DT*J1[i][k];
				tmpm2[i][k]=-0.5*DT*J2[i][k];
				if(i==k){
					
					tmpm1[i][k] = -1 + tmpm1[i][k];
					tmpm2[i][k] = 1 + tmpm2[i][k];
				}
			}
		}
		for(i=0;i<NX;i++){
			tmpd = x2[i]-x1[i]-0.5*DT*(f1[i]+f2[i]);
			action = action + Rf*tmpd*tmpd;
			for(k=0;k<NX;k++){
				
				grad_m[j][k] = grad_m[j][k] + 2*Rf*tmpd * tmpm1[i][k];
				grad_m[j+1][k] = grad_m[j+1][k] + 2*Rf*tmpd * tmpm2[i][k];
			}
		}
	}
	
	for(i=0;i<NT;i++){
		for(j=0;j<NX;j++){
			grad[NX*i+j] = grad_m[i][j];
		}
	}
}

void TDaction_grad(const real_1d_array &x, double &action, real_1d_array &grad, void *ptr) {
	real_2d_array XX, grad_m, test;
   	XX.setlength(NT,NX);
	grad_m.setlength(NT,NX);
	test.setlength(NT,NX);
	int i,j,k;
	for(i=0;i<NT;i++){
		for(j=0;j<NX;j++){
			XX[i][j] = x[NX*i+j];
		}
	}
	//reshapevector2matrix(x,XX);

	//URI
	// Extract beta, Time delay info from ptr:
	int beta = *(int *)ptr;

	double Rm=2, Rf=0.005*pow(2,beta),tmpd;
	real_1d_array x1,x2,f1,f2,tmpa1,tmpa2;
	x1.setlength(NX);
	x2.setlength(NX);
	f1.setlength(NX);
	f2.setlength(NX);
	tmpa1.setlength(NX);
	tmpa2.setlength(NX);
	real_2d_array J1, J2, tmpm1, tmpm2;
	J1.setlength(NX,NX);
	J2.setlength(NX,NX);
	tmpm1.setlength(NX,NX);
	tmpm2.setlength(NX,NX);

	action = 0;
	for(i=0;i<NT;i++)
		for(j=0;j<NX;j++)
			grad_m[i][j]=0;

	// I haven't changed anything, but this is where one could
	// generalize which variables are measured
	for(i=0;i<NMEA;i++){
		for(j=0;j<NT;j++){
			
			tmpd = (XX[j][2*i]-Ydata[j][2*i]);	
		
			action = action + Rm*tmpd*tmpd;
			grad_m[j][2*i] = grad_m[j][2*i] + 2*Rm*tmpd;
		}
	}
	
	for(j=0;j<(NT-1);j++){
		slice(XX, j, x1);
		slice(XX, j+1, x2);
		func_origin(x1, f1);
		func_origin(x2, f2);
		
		func_DF(x1,J1);
		func_DF(x2,J2);
		for(i=0;i<NX;i++){
			for(k=0;k<NX;k++){
				tmpm1[i][k]=-0.5*DT*J1[i][k];
				tmpm2[i][k]=-0.5*DT*J2[i][k];
				if(i==k){
					
					tmpm1[i][k] = -1 + tmpm1[i][k];
					tmpm2[i][k] = 1 + tmpm2[i][k];
				}
			}
		}
		for(i=0;i<NX;i++){
			tmpd = x2[i]-x1[i]-0.5*DT*(f1[i]+f2[i]);
			action = action + Rf*tmpd*tmpd;
			for(k=0;k<NX;k++){
				
				grad_m[j][k] = grad_m[j][k] + 2*Rf*tmpd * tmpm1[i][k];
				grad_m[j+1][k] = grad_m[j+1][k] + 2*Rf*tmpd * tmpm2[i][k];
			}
		}
	}
	
	for(i=0;i<NT;i++){
		for(j=0;j<NX;j++){
			grad[NX*i+j] = grad_m[i][j];
		}
	}
	
	// Time-Delay stuff starts here --URI

	//Safest to skip entirely, if NTD==0 i think
	if(NTD!=0){
	  double Rtd = 1.0/(1.0/Rm+1.0/Rf);
//	  if(beta == 0)
//	    cout << " Rtd=" << Rtd << " ";

	  real_1d_array delayedMap[NTD];
	  real_2d_array chain, dftmp;
	  real_2d_array delayedDF[NTD];
	  real_1d_array maptmp0, maptmp1;

	  for(i=0; i<NTD; i++){
	    delayedMap[i].setlength(NX);
	    delayedDF[i].setlength(NX,NX);
	  }

	  chain.setlength(NX,NX);
	  dftmp.setlength(NX,NX);
	  maptmp0.setlength(NX);
	  maptmp1.setlength(NX);
	  
	  real_1d_array xcurrent;
	  xcurrent.setlength(NX);

	  for(i=0; i<NT-taus[NTD-1]; i++){

	    slice(XX, i, xcurrent);

	    discF(xcurrent,maptmp0);
	    discDF(xcurrent, chain);

	    int count = 0;
	    int tau;

	    // CHECK THIS FOR BUG!!!!
	    for(tau=1; tau<=taus[NTD-1]; tau++){

	      discF(maptmp0, maptmp1);

	      // Multiply next chain-ruled DF-matrix AFTER storing
	      // it. For a given time delay, we want the DF evaluated
	      // up to DF(f^tau-1(x)) Ex: for Tau = 1, only want
	      // df(x)/dx, not df(f)/dx * df(x)/dx
	      if(count >= NTD)
		throw "COUNT too big, TAUS out of bounds";
	      if(tau==taus[count]){
		delayedDF[count] = chain;
		delayedMap[count] = maptmp0;
		count++;
	      }


	      //Want grad dA/dx(n) = (f^T(x)-y(n+T))*(DF(f^(T-1))*DF(F^(t-2))*...DF(x)
	      discDF(maptmp0, dftmp);
	      simple_mmult(dftmp,chain,chain);
	      maptmp0 = maptmp1;

	    } 
	  
	    for(count=0; count<NTD; count++){
	      real_2d_array dfchain = delayedDF[count];

	      for(j=0; j<NMEA; j++){	      	      
		int idx = measIdx[j];
		// f^tau(x(n)) - y(n+tau) ; Measured indices only
		if(count>=NTD)
		  throw("ERROR! COUNT TOO big!");
		if(idx>=NX)
		  throw("ERROR! idx TOO big!");

		if(i+taus[count]>=NT)
		     throw("AHHHHHH");
		double dyTD =  delayedMap[count][idx] - Ydata[i+taus[count]][idx];
		action += Rtd*dyTD*dyTD;
		//CHECK!
		for(k=0; k<NX; k++){	   
		  // grad_td[i][k] += 2*Rtd*dyTD*dfchain[idx][k]
		  // grad[NX*i+k] = grad_td[i][k]

		  grad[NX*i+k] += 2*Rtd*dyTD*dfchain[idx][k];
		}
	      }
	    }

   	  
	  }
	}

	

}

int main(int argc, char **argv)
{
  printf("HELLO!");
       // Set measured indices --URI
       for(int i=0; i<NMEA; i++)
	 measIdx[i] = 2*i;

    //
    // using LBFGS method.
    //
	Ydata.setlength(NT,NX);
	real_1d_array X0,grad_a;
	X0.setlength(NT*NX);
	grad_a.setlength(NT*NX);
	real_2d_array result;
	result.setlength(NBETA,(3+NT*NX));
	double act;
    	int i,j;
	int beta = 1, ipath;

	void *ptr;

    	double epsg = 1e-8;
    	double epsf = 1e-8;
    	double epsx = 1e-8;
	FILE *fp_output;
	char filename[50];

   	ae_int_t maxits = 10000;

	readdata(Ydata);
	
	minlbfgsstate state;
	minlbfgsreport rep;

	// Make Initial paths - Load Later. URI
	if(generate_paths == true){
	  
	  ofstream initpaths("initpaths.txt");
	  for(ipath=0;ipath<NPATH;ipath++){
	    for(i=0;i<NX*NT;i++) 
	      X0[i]=20*randomreal()-10;

	    if(initpaths.is_open()){
	      for(i=0;i<NX*NT;i++) 
		initpaths << X0[i] << " ";
	      initpaths << endl;	      
	    }
	    else{
	      printf("ERROR FILE WAS NOT OPENED");
	    }
	  }	  
	  initpaths.close();
	  cout << "Last entry in initpaths.txt: " << X0[NT*NX-1];
	}


	//load init paths file --Needed for all loop iterations
	// either loadpaths or lastpath will be used depending on BETASTART	

	//loadpaths must be declared before ipath loop, because all
	//paths are contained in the loadpaths file, so it has to
	//persist through each iteration.
	ifstream loadpaths("initpaths.txt");

	int nans = 0;       

	for(ipath=0;ipath<NPATH;ipath++){
	     string taustr("Ntd%d_");

	     for(i=0; i<NTD;i++)
		  taustr = taustr + std::to_string(taus[i]) + "-";

	     string temp("pathNoise/D%d_M%d_PATH%d_"+taustr+"dt%e.dat");
	     sprintf(filename, temp.c_str(), NX,NMEA,ipath,NTD,DT);

		//lastpath is the filename used to continue from a
		//non-zero BETASTART. It must occur within ipath loop,
		//because each path is contained in different file
		ifstream lastpath(filename);

		//Make BETASTART variable to continue old thing
		//load in initial paths --URI
		if(BETASTART ==0){
		  fp_output = fopen(filename,"w"); 		
		  if(loadpaths.is_open()){
		    for(i=0;i<NX*NT;i++)
		      loadpaths >> X0[i];			  
		    cout << "Last entry loaded from initpaths.txt: " << X0[NT*NX-1];
		  }
		  else{
		    printf("ERROR INIT PATHS NOT FOUND!");
		  }
		}
		else{
		  fp_output = fopen(filename,"a");
		  if(lastpath.is_open()){			     
		    int junkBeta, junkTerm;
		    double junkAct;

		    // Throwaway first 3 entries in line
		    string line;
		    for(i=0; i<BETASTART-1;i++)
		      getline(lastpath, line);

		    lastpath >> junkBeta >> junkTerm >> junkAct;
		    
		    for(i=0;i<NX*NT;i++)
		      lastpath >> X0[i];			  
		  }
		  else{
		    printf("EXISTING PATHFILE NOT FOUND!");
		  }
		}
		
			  
		for(beta=BETASTART;beta<NBETA;beta++){

		        printf("ipath=%d beta=%d\n", ipath, beta); 
			
			// set beta - URI
			ptr = &beta;

			minlbfgscreate(1, X0, state);

			minlbfgssetcond(state, epsg, epsf, epsx, maxits);

			minlbfgsoptimize(state, TDaction_grad, NULL, ptr);
			minlbfgsresults(state, X0, rep);
			TDaction_grad(X0, act, grad_a, ptr);
			//printf("run here\n");
			fprintf(fp_output, "%d %d %e ", beta, int(rep.terminationtype), act);
			printf("Min A0 = %e \n", act);			
			for(i=0;i<NX*NT;i++)
				fprintf(fp_output,"%e ", X0[i]);
			fprintf(fp_output,"\n");

			// Sometimes TD returns NaN Action values... URI
			if(isnan(act)){			     
			  nans += 1;
			  break;
			}
		}
		fclose(fp_output);
		if(lastpath.is_open())
		  lastpath.close();
	}
	if(loadpaths.is_open())
	  loadpaths.close();
	printf("Number of NaN Paths = %d", nans);
	return 0;
}
