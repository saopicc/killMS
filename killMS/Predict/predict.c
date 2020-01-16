// killMS, a package for calibration in radio interferometry.
// Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
// SKA South Africa, Rhodes University
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

/* A file to test imorting C modules for handling arrays to Python */
//#define NPY_NO_DEPRECATED_API	NPY_1_8_API_VERSION
#define NPY_NO_DEPRECATED_API	NPY_1_8_API_VERSION

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include "complex.h"
#include "predict.h"
#include <assert.h>
#include <stdio.h>








/* ==== Set up the methods table ====================== */
static PyMethodDef module_functions[] = {
	{"predictJones2_Gauss", predictJones2_Gauss, METH_VARARGS},
	{"ApplyJones", ApplyJones, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef _mod = {
    PyModuleDef_HEAD_INIT,
    "predict3x",
    "predict3x",
    -1,  
    module_functions,
    NULL,
    NULL,
    NULL,
    NULL
  };
  PyMODINIT_FUNC PyInit_predict3x(void) {
    PyObject* m = PyModule_Create(&_mod);
    import_array();
    return m;
  }
#else
    void initpredict27()  {
    (void) Py_InitModule("predict27", module_functions);
    import_array();  // Must be present for NumPy.  Called first after above line.
    }
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////


static PyObject *predict(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyObject *LSM, *LUVWSpeed, *LFreqs,*LSmearMode;
  PyArrayObject *NpVisIn, *NpUVWin, *matout;
  float *p_l,*p_m,*p_alpha,*p_Flux, *WaveL;
  int AllowChanEquidistant;
  double *UVWin;
  int nrow,npol,nsources,i,dim[2];
  
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!i",
			//&ObjVisIn,
			&PyArray_Type, &NpVisIn, 
			&PyArray_Type, &NpUVWin, 
			&PyList_Type, &LFreqs,
			&PyList_Type, &LSM,
			&PyList_Type, &LUVWSpeed,
			&PyList_Type, &LSmearMode,
			&AllowChanEquidistant))  return NULL;
  
  //NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX64, 0, 3);
  float complex* VisIn=(float complex*) PyArray_DATA(NpVisIn);
  
  PyArrayObject *Np_l;
  Np_l = (PyArrayObject *) PyList_GetItem(LSM, 0);
  PyArrayObject *Np_m;
  Np_m = (PyArrayObject *) PyList_GetItem(LSM, 1);
  PyArrayObject *Np_I;
  Np_I = (PyArrayObject *) PyList_GetItem(LSM, 2);
  
  PyArrayObject *NpWaveL;
  NpWaveL= (PyArrayObject *)  PyList_GetItem(LFreqs, 0);
  PyArrayObject *NpFreqs;
  NpFreqs= (PyArrayObject *)  PyList_GetItem(LFreqs, 1);
  PyArrayObject *NpDFreqs;
  NpDFreqs= (PyArrayObject *) PyList_GetItem(LFreqs, 2);
  float *p_DFreqs=(float *) PyArray_DATA(NpDFreqs);
  float *p_Freqs=(float *) PyArray_DATA(NpFreqs);



  PyArrayObject *NpUVW_dt;
  NpUVW_dt= (PyArrayObject *) (PyList_GetItem(LUVWSpeed, 0));
  //PyArrayObject *NpDFreqs;
  PyObject *_DT  = PyList_GetItem(LUVWSpeed, 1);
  float DT=(float) PyFloat_AsDouble(_DT);
  float *UVW_dt=(float *) PyArray_DATA(NpUVW_dt);

  PyObject *_FSmear  = PyList_GetItem(LSmearMode, 0);
  int FSmear=(int) PyFloat_AsDouble(_FSmear);
  PyObject *_TSmear  = PyList_GetItem(LSmearMode, 1);
  int TSmear=(int) PyFloat_AsDouble(_TSmear);





  UVWin=(double *) PyArray_DATA(NpUVWin);

  p_l=(float *) PyArray_DATA(Np_l);
  p_m=(float *) PyArray_DATA(Np_m);
  p_Flux=(float *) PyArray_DATA(Np_I);

  WaveL=(float *) PyArray_DATA(NpWaveL);
  
  int ch,dd,nchan,ndir;

  npy_intp *Cat_DIMS=PyArray_DIMS(Np_l);
  ndir=Cat_DIMS[0];
  
  npy_intp *UVW_DIMS=PyArray_DIMS(NpVisIn);
  nrow=UVW_DIMS[0];
  nchan=UVW_DIMS[1];


  int ChanEquidistant=0;
  /* if(nchan>2){ */
  /*   ChanEquidistant=1; */
  /*   float dFChan0=p_Freqs[1]-p_Freqs[0]; */
  /*   for(ch=0; ch<(nchan-1); ch++){ */
  /*     float df=abs(p_Freqs[ch+1]-p_Freqs[ch]); */
  /*     float ddf=abs(1.-df/dFChan0); */
  /*     //printf("df,ddf %i %f %f\n",ch,df,ddf); */
  /*     if(ddf>1e-3){ChanEquidistant=0;} */
  /*   } */
  /* } */
  /* if(AllowChanEquidistant==0){ */
  /*   ChanEquidistant=0; */
  /* } */

  ChanEquidistant=AllowChanEquidistant;
  //printf("ChanEquidistant %i\n",ChanEquidistant);
  

  /* Get the dimensions of the input */
  
  
  /* Do the calculation. */
  float phase,l,m,n,u,v,w;
  float complex c0,result;
  float C=299792456.;
  float PI=3.141592;
  c0=2.*PI*I;
  float complex *p0;
  double *p1;
  p0=VisIn;
  p1=UVWin;
  float complex c1[nchan];
  for(ch=0;ch<nchan;ch++){
    c1[ch]=c0/WaveL[ch];
    //printf("chan %f,%f: l=%f\n",creal(c1[ch]),cimag(c1[ch]),WaveL[ch]);
  }

  //float dnu=C/WaveL[0]-C/WaveL[nchan-1];
  float PI_C=PI/C;
  float phi,du,dv,dw,dphase;
  float complex Kernel;
  double complex dKernel;
  for(dd=0;dd<ndir;dd++){
    l=p_l[dd];
    m=p_m[dd];
    n=sqrt(1.-l*l-m*m)-1.;
    //    printf("dd: %i/%i nchan=%i nrow=%i (l,m)=(%f,%f)\n",dd,ndir,nchan,nrow,l,m);
    //printf("l,m: %f %f %f\n",l,m,n);
    //printf("\n");
    VisIn=p0;
    UVWin=p1;
    for ( i=0; i<nrow; i++)  {
  	phase=(*UVWin++)*l;
  	//printf("cc %f \n",phase);
  	phase+=(*UVWin++)*m;
  	//printf("cc %f \n",phase);
  	phase+=(*UVWin++)*n;
  	//printf("cc %f \n",phase);

  	for(ch=0;ch<nchan;ch++){

  	  if(ChanEquidistant==0){
  	    Kernel=cexp(phase*c1[ch]);
  	  }else{
  	    if(ch==0){
  	      Kernel=cexp(phase*c1[ch]);
  	      dKernel=cexp(phase*(c1[ch+1]-c1[ch]));
  	    }
  	    else{
  	      Kernel*=dKernel;
  	    }
  	  }

  	  //printf("ch: %i %f\n",ch,WaveL[ch]);
  	  result=p_Flux[dd*nchan+ch]*Kernel;
  	  if(FSmear==1){
  	    phi=PI*(p_DFreqs[ch]/C)*phase;
  	    if(phi!=0.){
  	      phi=sin(phi)/(phi);
  	      result*=phi;
  	    };
  	  };
  	  if(TSmear==1){
	    
  	    du=UVW_dt[3*i]*l;
  	    dv=UVW_dt[3*i+1]*m;
  	    dw=UVW_dt[3*i+2]*n;
  	    dphase=(du+dv+dw)*DT;
  	    phi=PI*(p_Freqs[ch]/C)*dphase;
  	    //printf("phi = %f\n",phi);
  	    //printf("dphase = %f\n",dphase);
  	    if(phi!=0.){
  	      phi=sin(phi)/(phi);
  	      result*=phi;
  	    };
  	  };


  	  //printf("\n");
  	  *VisIn++   += result;
  	  VisIn++;VisIn++;
  	  *VisIn++   += result;
  	}
	
    }
  }

  //return Py_None;  
  Py_INCREF(Py_None);
  return Py_None;
  //  return PyArray_Return(NpVisIn);
}







//////////////////////////////////////  PREDICT JONES





/////////////////////////////



/////////////////////////////////////////////////////////////////////////

static PyObject *predictJones2_Gauss(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyObject *LSM, *LUVWSpeed, *LFreqs,*LSmearMode, *LJones, *LExp,*LSinc;
  PyArrayObject *NpVisIn, *NpUVWin, *matout;
  float *p_alpha,*p_Flux;
  
  double *p_l,*p_m,*UVWin, *WaveL;
  int nrow,npol,nsources,i,dim[2];
  int AllowChanEquidistant;
  
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iO!O!",
			//&ObjVisIn,
			&PyArray_Type, &NpVisIn, 
			&PyArray_Type, &NpUVWin, 
			&PyList_Type, &LFreqs,
			&PyList_Type, &LSM,
			&PyList_Type, &LUVWSpeed,
			&PyList_Type, &LSmearMode,
			//&PyList_Type, &LJones,
			&AllowChanEquidistant,		       
			&PyList_Type, &LExp,
			&PyList_Type, &LSinc
			))  return NULL;
  


  float complex* VisIn=(float complex*) PyArray_DATA(NpVisIn);
  float complex* ThisVis;
  float complex VisCorr[4]={0};

  PyArrayObject *Np_l;
  Np_l = (PyArrayObject *) (PyList_GetItem(LSM, 0));
  PyArrayObject *Np_m;
  Np_m = (PyArrayObject *) (PyList_GetItem(LSM, 1));
  PyArrayObject *Np_I;
  Np_I = (PyArrayObject *) (PyList_GetItem(LSM, 2));

  //===================
  // Gaussian Pars
  PyArrayObject *Np_Gmin;
  Np_Gmin = (PyArrayObject *) (PyList_GetItem(LSM, 3));
  PyArrayObject *Np_Gmaj;
  Np_Gmaj = (PyArrayObject *) (PyList_GetItem(LSM, 4));
  PyArrayObject *Np_GPA;
  Np_GPA = (PyArrayObject *) (PyList_GetItem(LSM, 5));

  float *p_Gmin=(float *) PyArray_DATA(Np_Gmin);
  float *p_Gmaj=(float *) PyArray_DATA(Np_Gmaj);
  float *p_GPA=(float *) PyArray_DATA(Np_GPA);
  //===================
  // Exponential table: exp(-x)
  PyArrayObject *Np_Exp;
  Np_Exp = (PyArrayObject *) (PyList_GetItem(LExp, 0));
  float *p_Exp=(float *) PyArray_DATA(Np_Exp);


  npy_intp *Np_Exp_DIMS=PyArray_DIMS(Np_Exp);
  int NmaxExp=Np_Exp_DIMS[0];
  
  PyObject *_FStepExp= PyList_GetItem(LExp, 1);
  float StepExp=(float) (PyFloat_AsDouble(_FStepExp));
  
  //===================
  // Sinc table: sin(x)/x
  PyArrayObject *Np_Sinc;
  Np_Sinc = (PyArrayObject *) (PyList_GetItem(LSinc, 0));
  float *p_Sinc=(float *) PyArray_DATA(Np_Sinc);


  npy_intp *Np_Sinc_DIMS=PyArray_DIMS(Np_Sinc);
  int NmaxSinc=Np_Sinc_DIMS[0];
  PyObject *_FStepSinc= PyList_GetItem(LSinc, 1);
  float StepSinc=(float) (PyFloat_AsDouble(_FStepSinc));
  
  //===================

  
  PyArrayObject *NpWaveL;
  NpWaveL= (PyArrayObject *)  (PyList_GetItem(LFreqs, 0));
  PyArrayObject *NpFreqs;
  NpFreqs= (PyArrayObject *)  (PyList_GetItem(LFreqs, 1));
  PyArrayObject *NpDFreqs;
  NpDFreqs= (PyArrayObject *) (PyList_GetItem(LFreqs, 2));

  float *p_DFreqs=(float *) PyArray_DATA(NpDFreqs);
  float *p_Freqs=(float *) PyArray_DATA(NpFreqs);

  PyArrayObject *NpUVW_dt;
  NpUVW_dt= (PyArrayObject *) (PyList_GetItem(LUVWSpeed, 0));
  //PyArrayObject *NpDFreqs;
  PyObject *_DT  = PyList_GetItem(LUVWSpeed, 1);
  float DT=(float) PyFloat_AsDouble(_DT);
  double *UVW_dt=(double *) PyArray_DATA(NpUVW_dt);
  
  PyObject *_FSmear  = PyList_GetItem(LSmearMode, 0);
  int FSmear=(int) PyFloat_AsDouble(_FSmear);
  PyObject *_TSmear  = PyList_GetItem(LSmearMode, 1);
  int TSmear=(int) PyFloat_AsDouble(_TSmear);





  UVWin=(double *) PyArray_DATA(NpUVWin);
  p_l=(double *) PyArray_DATA(Np_l);
  p_m=(double *) PyArray_DATA(Np_m);
  p_Flux=(float *) PyArray_DATA(Np_I);
  
  //printf("l=%f",((float)p_l[0]));

  WaveL=(double *) PyArray_DATA(NpWaveL);
  
  int ch,dd,nchan,ndir;

  npy_intp *NpVisIn_DIMS=PyArray_DIMS(NpVisIn);
  nrow=NpVisIn_DIMS[0];
  nchan=NpVisIn_DIMS[1];


  int ChanEquidistant=0;

  ChanEquidistant=AllowChanEquidistant;

  npy_intp *Np_l_DIMS=PyArray_DIMS(Np_l);
  ndir=Np_l_DIMS[0];
  //printf("ndir=%i ",ndir);
  
  /* Get the dimensions of the input */
  
  
  /* Do the calculation. */
  double phase,l,m,n,u,v,w;
  
  double complex c0;
  float complex result;
  float C=299792458.;
  float PI=3.141592;
  c0=2.*PI*I;
  float complex *p0;
  double *p1;
  p0=VisIn;
  p1=UVWin;

  double complex c1[nchan];
  float complex c2[nchan];
  for(ch=0;ch<nchan;ch++){
    c1[ch]=c0/WaveL[ch];
    c2[ch]=2*(PI/WaveL[ch])*(PI/WaveL[ch]);
    //printf("chan %f,%f: l=%f\n",creal(c1[ch]),cimag(c1[ch]),WaveL[ch]);
  }

  //float dnu=C/WaveL[0]-C/WaveL[nchan-1];
  float PI_C=PI/C;
  float phi,decorr,du,dv,dw,dphase;

  float complex J0[4]={0},J1[4]={0},J0inv[4]={0},J1H[4]={0},J1Hinv[4]={0},JJ[4]={0};
  
  int ApplyJones=1;
  int irow;

  complex Kernel;
  complex dKernel;
  int ThisSourceType;

  float SminCos,SminSin,SmajCos,SmajSin;

  for(dd=0;dd<ndir;dd++){
    l=p_l[dd];
    //printf("l=%f",((float)l));

    m=p_m[dd];
    n=sqrt(1.-l*l-m*m)-1.;
    //    printf("dd: %i/%i nchan=%i nrow=%i (l,m)=(%f,%f)\n",dd,ndir,nchan,nrow,l,m);
    //printf("l,m: %f %f %f\n",l,m,n);

    //printf("\n");
    VisIn=p0;
    UVWin=p1;
    float ang=p_GPA[dd];
    float SigMaj=p_Gmaj[dd];
    float SigMin=p_Gmin[dd];
    //printf("%i:\n",dd);
    //printf("%f %f %f\n",SigMin,SigMaj,ang);

    ThisSourceType=(SigMaj!=0.);

    if(ThisSourceType==1){
      SminCos=SigMin*cos(ang);
      SminSin=SigMin*sin(ang);
      SmajCos=SigMaj*cos(ang);
      SmajSin=SigMaj*sin(ang);
    }
    
    float G0,G1,UVsq,UVsq_ch,FGauss;
    for ( irow=0; irow<nrow; irow++)  {
    //for ( irow=1997; irow<1999; irow++)  {
      //printf("\n");

      float U=*UVWin;
      phase=(*UVWin++)*l;
      float V=*UVWin;
      //printf("cc0 %f \n",phase);
      phase+=(*UVWin++)*m;
      //printf("cc1 %f \n",phase);
      phase+=(*UVWin++)*n;
      //printf("phase %f \n",phase);

      if(ThisSourceType==1){
    	G0=(U*SminCos-V*SminSin);
    	G1=(U*SmajSin+V*SmajCos);
    	UVsq=(G0*G0+G1*G1);
      }

      for(ch=0;ch<nchan;ch++){
    	  ThisVis=VisIn+irow*nchan*4+ch*4;


    	  if(ChanEquidistant==0){
    	    Kernel=cexp((phase*c1[ch]));
    	  }else{
    	    if(ch==0){
    	      Kernel=cexp((float complex)(phase*c1[ch]));
    	      dKernel=cexp((float complex)(phase*(c1[ch+1]-c1[ch])));
    	    }
    	    else{
    	      Kernel*=dKernel;
    	    }
    	  }

    	  if(ThisSourceType==1){
    	    UVsq_ch=c2[ch]*UVsq;
    	    FGauss=p_Flux[dd*nchan+ch]*GiveFunc(UVsq_ch,p_Exp,StepExp, NmaxExp);
    	  }
    	  else{
    	    FGauss=p_Flux[dd*nchan+ch];
    	  }

    	  result=FGauss*((float complex)Kernel);
    	  //printf("result (%f,%f) \n",creal(result),cimag(result));

    	  if(FSmear==1){
    	    //phi=PI*PI_C*p_DFreqs[ch]*phase;
    	    phi=PI*(p_DFreqs[ch]/C)*phase;
	    
    	    if(phi!=0.){
	      //phi=1.471034;
    	      decorr=(float)sin((double)phi)/((double)phi);
	      float decorr2=GiveFunc(fabs(phi),p_Sinc,StepSinc, NmaxSinc);
	      //printf("%f %f %f\n",phi,decorr,decorr2);
    	      result*=decorr;
    	    };
    	  };
    	  if(TSmear==1){
	    
    	    du=UVW_dt[3*irow]*l;
    	    dv=UVW_dt[3*irow+1]*m;
    	    dw=UVW_dt[3*irow+2]*n;
    	    dphase=(du+dv+dw)*DT;
    	    //phi=PI*PI_C*p_Freqs[ch]*dphase;
    	    phi=PI*(p_Freqs[ch]/C)*dphase;
    	    //printf("phi = %f\n",phi);
    	    //printf("dphase = %f\n",dphase);
    	    if(phi!=0.){
	      decorr=sin(phi)/(phi);
    	      //decorr=GiveFunc(phi,p_Sinc,StepSinc, NmaxSinc);
    	      result*=decorr;
    	    };
    	  };


    	  //printf("\n");
	  
    	  ThisVis[0]   += result;
    	  ThisVis[3]   += result;
    	  //printf("(%f,%f)\n",creal(ThisVis[0]),cimag(ThisVis[0]));


    	}
	
    }
  }






  Py_INCREF(Py_None);
  return Py_None;
  ////return Py_None;  
  //return PyArray_Return(NpVisIn);
}



/////////////////////////////////////////////////////////////////////////

static PyObject *ApplyJones(PyObject *self, PyObject *args)
{
  PyObject *LJones;
  PyArrayObject *NpVisIn;
  int nrow,npol,nsources,i,dim[2];
  
  if (!PyArg_ParseTuple(args, "O!O!",
			&PyArray_Type, &NpVisIn, 
			&PyList_Type, &LJones))  return NULL;
  

  int LengthJonesList=PyList_Size(LJones);
  int DoApplyJones=0;
  PyArrayObject *npJonesMatrices, *npTimeMappingJonesMatrices, *npFreqMappingJonesMatrices, *npA0, *npA1, *npJonesIDIR, *npCoefsInterp,*npModeInterpolation;
  float complex* ptrJonesMatrices;
  int *ptrTimeMappingJonesMatrices,*ptrFreqMappingJonesMatrices,*ptrA0,*ptrA1,*ptrJonesIDIR;
  float *ptrCoefsInterp;
  int i_dir;
  int nd_Jones,na_Jones,nch_Jones,nt_Jones;
  
  int JonesDims[4];
  int ModeInterpolation=1;
  int *ptrModeInterpolation;

  npTimeMappingJonesMatrices  = (PyArrayObject *) (PyList_GetItem(LJones, 0));
  ptrTimeMappingJonesMatrices=(int *) PyArray_DATA(npTimeMappingJonesMatrices);

  npA0 = (PyArrayObject *) (PyList_GetItem(LJones, 1));
  ptrA0=(int *) PyArray_DATA(npA0);

  npA1= (PyArrayObject *) (PyList_GetItem(LJones, 2));
  ptrA1=(int *) PyArray_DATA(npA1);
 
      
  // (nt,nd,na,1,2,2)
  npJonesMatrices = (PyArrayObject *) (PyList_GetItem(LJones, 3));
  ptrJonesMatrices=(float complex *) PyArray_DATA(npJonesMatrices);
  
  npy_intp *npJonesMatrices_DIMS=PyArray_DIMS(npJonesMatrices);
  nt_Jones=npJonesMatrices_DIMS[0];
  nd_Jones=npJonesMatrices_DIMS[1];
  na_Jones=npJonesMatrices_DIMS[2];
  nch_Jones=npJonesMatrices_DIMS[3];
  JonesDims[0]=nt_Jones;
  JonesDims[1]=nd_Jones;
  JonesDims[2]=na_Jones;
  JonesDims[3]=nch_Jones;

  npFreqMappingJonesMatrices  = (PyArrayObject *) (PyList_GetItem(LJones, 4));
  ptrFreqMappingJonesMatrices=(int *) PyArray_DATA(npFreqMappingJonesMatrices);

  
  PyObject *_IDIR  = PyList_GetItem(LJones, 5);
  i_dir=(int) PyFloat_AsDouble(_IDIR);
  // printf("idir=%i\n",i_dir);

  float complex* VisIn=(float complex *) PyArray_DATA(NpVisIn);
  float complex* ThisVis;
  float complex VisCorr[4]={0};
  
  int ch,dd,nchan,ndir;
  npy_intp *NpVisIn_DIMS=PyArray_DIMS(NpVisIn);
  nrow=NpVisIn_DIMS[0];
  nchan=NpVisIn_DIMS[1];

  float complex J0[4]={0},J1[4]={0},J0inv[4]={0},J1H[4]={0},J1Hinv[4]={0},JJ[4]={0};
  
  int irow;

  for ( irow=0; irow<nrow; irow++)  {
    //printf("irow = %i\n",irow);
    for(ch=0;ch<nchan;ch++){
      int i_t=ptrTimeMappingJonesMatrices[irow];
      int i_chJones=ptrFreqMappingJonesMatrices[ch];
      int i_ant0=ptrA0[irow];
      int i_ant1=ptrA1[irow];
      //printf("  [%i, %i, %i]: (t,dir,ch)_Jones = (%i, %i, %i)\n",ch,i_ant0,i_ant1, i_t,i_dir,i_chJones);
      GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant0, i_dir, i_chJones, J0);
      GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant1, i_dir, i_chJones, J1);
      
      MatH(J1,J1H);
      MatDot(J0,J1H,JJ);

      size_t off=irow*nchan*4+ch*4;
      ThisVis=VisIn+off;
      float complex result=ThisVis[0];
      
      ThisVis[0]   = result*JJ[0];
      ThisVis[1]   = result*JJ[1];
      ThisVis[2]   = result*JJ[2];
      ThisVis[3]   = result*JJ[3];
      
    }
  }
  

  Py_INCREF(Py_None);
  return Py_None;
}


