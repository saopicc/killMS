/* A file to test imorting C modules for handling arrays to Python */

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include "complex.h"
#include "predict.h"
#include <assert.h>
#include <stdio.h>

/* #### Globals #################################### */

/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */

/* ==== Set up the methods table ====================== */
static PyMethodDef predict_Methods[] = {
	{"predict", predict, METH_VARARGS},
	{"predictJones", predictJones, METH_VARARGS},
	{"GiveMaxCorr", GiveMaxCorr, METH_VARARGS},
	{"CorrVis", CorrVis, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void initpredict()  {
	(void) Py_InitModule("predict", predict_Methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}


void GiveJones(float complex *ptrJonesMatrices, int *JonesDims, float *ptrCoefs, int i_t, int i_ant0, int i_dir, float complex *Jout){
  int nd_Jones,na_Jones,nch_Jones;
  nd_Jones=JonesDims[1];
  na_Jones=JonesDims[2];
  nch_Jones=JonesDims[3];
  
  int ipol,idir;
  int offJ0=i_t*nd_Jones*na_Jones*nch_Jones*4
    +i_dir*na_Jones*nch_Jones*4
    +i_ant0*nch_Jones*4;
  for(ipol=0; ipol<4; ipol++){
    Jout[ipol]=*(ptrJonesMatrices+offJ0+ipol);
  }
  
}


static PyObject *CorrVis(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn, *ObjVisCorr;
  PyObject *LSM, *LJones;
  PyArrayObject *NpVisIn,*NpVisCorr, *NpUVWin, *matout;
  float *p_l,*p_m,*p_alpha,*p_Flux, *WaveL;

  int nrow,npol,nsources,i,dim[2];
  
  if (!PyArg_ParseTuple(args, "OOO!",
			&ObjVisIn,
			&ObjVisCorr,
			&PyList_Type, &LJones))  return NULL;
  



  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX64, 0, 3);
  float complex* __restrict__ VisIn=p_complex64(NpVisIn);

  NpVisCorr = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisCorr, PyArray_COMPLEX64, 0, 3);
  float complex* __restrict__ visCorr=p_complex64(NpVisCorr);



  /* float complex *visCorr=malloc((nrow*nchan*4)*sizeof(float complex)); */
  /* memset(visCorr, 0, (nrow*nchan*4)*sizeof(float complex)); */
  /* npy_intp NpShape[3]; */
  /* NpShape[0]=nrow; */
  /* NpShape[1]=nchan; */
  /* NpShape[2]=4; */
  /* PyArrayObject * NpVisCorr = (PyArrayObject*)PyArray_SimpleNewFromData(3, NpShape, NPY_COMPLEX64, visCorr); */
  



  //////////////////////////////////////////////

  int LengthJonesList=PyList_Size(LJones);
  int DoApplyJones=0;
  PyArrayObject *npJonesMatrices, *npTimeMappingJonesMatrices, *npA0, *npA1, *npJonesIDIR, *npCoefsInterp,*npModeInterpolation;
  float complex* ptrJonesMatrices;
  int *ptrTimeMappingJonesMatrices,*ptrA0,*ptrA1,*ptrJonesIDIR;
  float *ptrCoefsInterp;
  int i_dir;
  int nd_Jones,na_Jones,nch_Jones,nt_Jones;
  
  int JonesDims[4];
  int ModeInterpolation=1;
  int *ptrModeInterpolation;

  npTimeMappingJonesMatrices  = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 0), PyArray_INT32, 0, 4);
  ptrTimeMappingJonesMatrices = p_int32(npTimeMappingJonesMatrices);

  npA0 = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 1), PyArray_INT32, 0, 4);
  ptrA0 = p_int32(npA0);

  npA1= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 2), PyArray_INT32, 0, 4);
  ptrA1=p_int32(npA1);
 
      
  // (nt,nd,na,1,2,2)
  npJonesMatrices = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 3), PyArray_COMPLEX64, 0, 6);
  ptrJonesMatrices=p_complex64(npJonesMatrices);
  nt_Jones=(int)npJonesMatrices->dimensions[0];
  nd_Jones=(int)npJonesMatrices->dimensions[1];
  na_Jones=(int)npJonesMatrices->dimensions[2];
  nch_Jones=(int)npJonesMatrices->dimensions[3];
  JonesDims[0]=nt_Jones;
  JonesDims[1]=nd_Jones;
  JonesDims[2]=na_Jones;
  JonesDims[3]=nch_Jones;

  PyObject *_IDIR  = PyList_GetItem(LJones, 4);
  i_dir=(int) PyFloat_AsDouble(_IDIR);

  /* ////////////////////////////////////////////// */





  
  int ch,dd,nchan,ndir;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];

  ndir=nd_Jones;

  /* Get the dimensions of the input */
  
  /* Make a new double matrix of same dims */
  //matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);

  
  
  /* Do the calculation. */
  float phase,l,m,n,u,v,w;
  float complex c0,result;
  float C=299792456.;
  float PI=3.141592;
  c0=2.*PI*I;
  float complex *p0;
  double *p1;
  p0=VisIn;

  int irow;


  //float dnu=C/WaveL[0]-C/WaveL[nchan-1];
  float PI_C=PI/C;
  float phi,du,dv,dw,dphase;
  float complex* __restrict__ visPtr_Uncorr;
  float complex visPtr[4];
  int ipol;


  



  VisIn=p0;
  float complex *visCorr22;
  
  for ( irow=0; irow<nrow; irow++)  {
    
    int i_t=ptrTimeMappingJonesMatrices[irow];
    int i_ant0=ptrA0[irow];
    int i_ant1=ptrA1[irow];
    //printf("%i %i %i %i | ",dd, i_t,i_ant0,i_ant1);
    
    float complex J0[4]={0},J1[4]={0},J0inv[4]={0},J1H[4]={0},J1Hinv[4]={0},JJ[4]={0};
    GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant0, i_dir, J0);
    GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant1, i_dir, J1);
    
    MatInv(J0,J0inv,0);
    MatH(J1,J1H);
    MatInv(J1H,J1Hinv,0);
    
    for(ch=0;ch<nchan;ch++){
      int doff = (irow * nchan + ch) * 4;
      
      visPtr_Uncorr  = VisIn  + doff;
      visCorr22 = visCorr + doff;
      MatDot(J0inv,visPtr_Uncorr,visCorr22);
      MatDot(visCorr22,J1Hinv,visCorr22);
      
    }

  }


  return PyArray_Return(NpVisCorr);
}



static PyObject *GiveMaxCorr(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyObject *LSM, *LJones;
  PyArrayObject *NpVisIn, *NpUVWin, *matout;
  float *p_l,*p_m,*p_alpha,*p_Flux, *WaveL;

  int nrow,npol,nsources,i,dim[2];
  
  if (!PyArg_ParseTuple(args, "OO!",
			&ObjVisIn,
			&PyList_Type, &LJones))  return NULL;
  



  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX64, 0, 3);
  float complex* __restrict__ VisIn=p_complex64(NpVisIn);

  



  //////////////////////////////////////////////

  int LengthJonesList=PyList_Size(LJones);
  int DoApplyJones=0;
  PyArrayObject *npJonesMatrices, *npTimeMappingJonesMatrices, *npA0, *npA1, *npJonesIDIR, *npCoefsInterp,*npModeInterpolation;
  float complex* ptrJonesMatrices;
  int *ptrTimeMappingJonesMatrices,*ptrA0,*ptrA1,*ptrJonesIDIR;
  float *ptrCoefsInterp;
  int i_dir;
  int nd_Jones,na_Jones,nch_Jones,nt_Jones;
  
  int JonesDims[4];
  int ModeInterpolation=1;
  int *ptrModeInterpolation;

  npTimeMappingJonesMatrices  = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 0), PyArray_INT32, 0, 4);
  ptrTimeMappingJonesMatrices = p_int32(npTimeMappingJonesMatrices);

  npA0 = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 1), PyArray_INT32, 0, 4);
  ptrA0 = p_int32(npA0);

  npA1= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 2), PyArray_INT32, 0, 4);
  ptrA1=p_int32(npA1);
 
      
  // (nt,nd,na,1,2,2)
  npJonesMatrices = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 3), PyArray_COMPLEX64, 0, 6);
  ptrJonesMatrices=p_complex64(npJonesMatrices);
  nt_Jones=(int)npJonesMatrices->dimensions[0];
  nd_Jones=(int)npJonesMatrices->dimensions[1];
  na_Jones=(int)npJonesMatrices->dimensions[2];
  nch_Jones=(int)npJonesMatrices->dimensions[3];
  JonesDims[0]=nt_Jones;
  JonesDims[1]=nd_Jones;
  JonesDims[2]=na_Jones;
  JonesDims[3]=nch_Jones;
  
  /* npJonesIDIR= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 4), PyArray_INT32, 0, 4); */
  /* ptrJonesIDIR=p_int32(npJonesIDIR); */
  /* i_dir=ptrJonesIDIR[0]; */

  /* ////////////////////////////////////////////// */





  
  int ch,dd,nchan,ndir;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];

  ndir=nd_Jones;

  /* Get the dimensions of the input */
  
  /* Make a new double matrix of same dims */
  //matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);

  
  
  /* Do the calculation. */
  float phase,l,m,n,u,v,w;
  float complex c0,result;
  float C=299792456.;
  float PI=3.141592;
  c0=2.*PI*I;
  float complex *p0;
  double *p1;
  p0=VisIn;

  int irow;


  //float dnu=C/WaveL[0]-C/WaveL[nchan-1];
  float PI_C=PI/C;
  float phi,du,dv,dw,dphase;
  float complex* __restrict__ visPtr_Uncorr;
  float complex visPtr[4];
  int ipol;

  float *visMax=malloc((nrow)*sizeof(float));
  memset(visMax, 0, (nrow)*sizeof(float));
  npy_intp NpShape[1];
  NpShape[0]=nrow;
  int npTypeF32=NPY_FLOAT32;
  PyArrayObject * NpVisMax = (PyArrayObject*)PyArray_SimpleNewFromData(1, NpShape, npTypeF32, visMax);
  
  float *StdDir=malloc((ndir)*sizeof(float));
  memset(StdDir, 0, (ndir)*sizeof(float));
  NpShape[0]=ndir;
  PyArrayObject * NpStdDir = (PyArrayObject*)PyArray_SimpleNewFromData(1, NpShape, npTypeF32, StdDir);

  float complex *SumDir=calloc(1,(ndir)*sizeof(float complex));
  float complex *SumDirSq=calloc(1,(ndir)*sizeof(float complex));

  for(dd=0;dd<ndir;dd++){
    int i_dir=dd;
    VisIn=p0;


    for ( irow=0; irow<nrow; irow++)  {

      int i_t=ptrTimeMappingJonesMatrices[irow];
      int i_ant0=ptrA0[irow];
      int i_ant1=ptrA1[irow];
      float complex J0[4]={0},J1[4]={0},J0inv[4]={0},J1H[4]={0},J1Hinv[4]={0},JJ[4]={0};
      GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant0, i_dir, J0);
      GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant1, i_dir, J1);
      
      MatInv(J0,J0inv,0);
      MatH(J1,J1H);
      MatInv(J1H,J1Hinv,0);

      for(ch=0;ch<nchan;ch++){
      	int doff = (irow * nchan + ch) * 4;

      	visPtr_Uncorr  = VisIn  + doff;
	    
      	MatDot(J0inv,visPtr_Uncorr,visPtr);
      	MatDot(visPtr,J1Hinv,visPtr);
	
      }
      float complex V=visPtr[0];
      SumDir[dd]+=V;
      float Amp=cabs(V);
      SumDirSq[dd]+=(Amp*Amp);
    }
  }
  
  for(dd=0;dd<ndir;dd++){
    SumDir[dd]/=nrow;
    SumDirSq[dd]/=nrow;
    StdDir[dd]=SumDirSq[dd]-SumDir[dd]*SumDir[dd];
  }

  /* for ( irow=0; irow<nrow; irow++)  { */
  /*     int i_t=ptrTimeMappingJonesMatrices[irow]; */
  /*     printf("%i %i \n",irow, i_t); */
  /* } */

  for(dd=0;dd<ndir;dd++){
    int i_dir=dd;
    VisIn=p0;


    for ( irow=0; irow<nrow; irow++)  {

      int i_t=ptrTimeMappingJonesMatrices[irow];
      int i_ant0=ptrA0[irow];
      int i_ant1=ptrA1[irow];
      //printf("%i %i %i %i | ",dd, i_t,i_ant0,i_ant1);
      
      float complex J0[4]={0},J1[4]={0},J0inv[4]={0},J1H[4]={0},J1Hinv[4]={0},JJ[4]={0};
      GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant0, i_dir, J0);
      GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant1, i_dir, J1);
      
      MatInv(J0,J0inv,0);
      MatH(J1,J1H);
      MatInv(J1H,J1Hinv,0);

      for(ch=0;ch<nchan;ch++){
      	int doff = (irow * nchan + ch) * 4;

      	visPtr_Uncorr  = VisIn  + doff;
      	for(ipol =0; ipol<4;ipol++){
      	  visPtr[ipol]=visPtr_Uncorr[ipol];
      	}
	    
      	MatDot(J0inv,visPtr_Uncorr,visPtr);
      	MatDot(visPtr,J1Hinv,visPtr);
	
      }
      float Amp=cabs(visPtr[0])/StdDir[dd];

      if(Amp>visMax[irow]){visMax[irow]=Amp;}
    }
  }

  //return Py_None;  
  return PyArray_Return(NpVisMax);
}





static PyObject *predict(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyObject *LSM, *LUVWSpeed, *LFreqs,*LSmearMode;
  PyArrayObject *NpVisIn, *NpUVWin, *matout;
  float *p_l,*p_m,*p_alpha,*p_Flux, *WaveL;
  int AllowChanEquidistant;
  double *UVWin;
  int nrow,npol,nsources,i,dim[2];
  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!i",
			&ObjVisIn,
			&PyArray_Type, &NpUVWin, 
			&PyList_Type, &LFreqs,
			&PyList_Type, &LSM,
			&PyList_Type, &LUVWSpeed,
			&PyList_Type, &LSmearMode,
			&AllowChanEquidistant))  return NULL;
  
  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX64, 0, 4);

  float complex* VisIn=p_complex64(NpVisIn);

  PyArrayObject *Np_l;
  Np_l = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LSM, 0), PyArray_FLOAT32, 0, 4);
  PyArrayObject *Np_m;
  Np_m = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LSM, 1), PyArray_FLOAT32, 0, 4);
  PyArrayObject *Np_I;
  Np_I = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LSM, 2), PyArray_FLOAT32, 0, 4);
  
  PyArrayObject *NpWaveL;
  NpWaveL= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LFreqs, 0), PyArray_FLOAT32, 0, 4);
  PyArrayObject *NpFreqs;
  NpFreqs= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LFreqs, 1), PyArray_FLOAT32, 0, 4);
  PyArrayObject *NpDFreqs;
  NpDFreqs= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LFreqs, 2), PyArray_FLOAT32, 0, 4);
  float *p_DFreqs=p_float32(NpDFreqs);
  float *p_Freqs=p_float32(NpFreqs);



  PyArrayObject *NpUVW_dt;
  NpUVW_dt= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LUVWSpeed, 0), PyArray_FLOAT32, 0, 4);
  //PyArrayObject *NpDFreqs;
  PyObject *_DT  = PyList_GetItem(LUVWSpeed, 1);
  float DT=(float) PyFloat_AsDouble(_DT);
  float *UVW_dt=p_float32(NpUVW_dt);

  PyObject *_FSmear  = PyList_GetItem(LSmearMode, 0);
  int FSmear=(int) PyFloat_AsDouble(_FSmear);
  PyObject *_TSmear  = PyList_GetItem(LSmearMode, 1);
  int TSmear=(int) PyFloat_AsDouble(_TSmear);





  UVWin=p_float64(NpUVWin);
  p_l=p_float32(Np_l);
  p_m=p_float32(Np_m);

  p_Flux=p_float32(Np_I);

  WaveL=p_float32(NpWaveL);
  
  int ch,dd,nchan,ndir;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];

  int ChanEquidistant=0;
  if(nchan>2){
    ChanEquidistant=1;
    float dFChan0=p_Freqs[1]-p_Freqs[0];
    for(ch=0; ch<(nchan-1); ch++){
      float df=abs(p_Freqs[ch+1]-p_Freqs[ch]);
      float ddf=abs(1.-df/dFChan0);
      printf("df,ddf %i %f %f\n",ch,df,ddf);
      if(ddf>1e-3){ChanEquidistant=0;}
    }
  }
  if(AllowChanEquidistant==0){
    ChanEquidistant=0;
  }
  printf("ChanEquidistant %i\n",ChanEquidistant);
  
  ndir=Np_l->dimensions[0];

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
  return PyArray_Return(NpVisIn);
}







//////////////////////////////////////  PREDICT JONES






static PyObject *predictJones(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyObject *LSM, *LUVWSpeed, *LFreqs,*LSmearMode, *LJones;
  PyArrayObject *NpVisIn, *NpUVWin, *matout;
  float *p_l,*p_m,*p_alpha,*p_Flux, *WaveL;

  double *UVWin;
  int nrow,npol,nsources,i,dim[2];
  int AllowChanEquidistant;
  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!O!i",
			&ObjVisIn,
			&PyArray_Type, &NpUVWin, 
			&PyList_Type, &LFreqs,
			&PyList_Type, &LSM,
			&PyList_Type, &LUVWSpeed,
			&PyList_Type, &LSmearMode,
			&PyList_Type, &LJones,
			&AllowChanEquidistant))  return NULL;
  

  //////////////////////////////////////////////

  int LengthJonesList=PyList_Size(LJones);
  int DoApplyJones=0;
  PyArrayObject *npJonesMatrices, *npTimeMappingJonesMatrices, *npA0, *npA1, *npJonesIDIR, *npCoefsInterp,*npModeInterpolation;
  float complex* ptrJonesMatrices;
  int *ptrTimeMappingJonesMatrices,*ptrA0,*ptrA1,*ptrJonesIDIR;
  float *ptrCoefsInterp;
  int i_dir;
  int nd_Jones,na_Jones,nch_Jones,nt_Jones;
  
  int JonesDims[4];
  int ModeInterpolation=1;
  int *ptrModeInterpolation;

  npTimeMappingJonesMatrices  = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 0), PyArray_INT32, 0, 4);
  ptrTimeMappingJonesMatrices = p_int32(npTimeMappingJonesMatrices);

  npA0 = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 1), PyArray_INT32, 0, 4);
  ptrA0 = p_int32(npA0);

  npA1= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 2), PyArray_INT32, 0, 4);
  ptrA1=p_int32(npA1);
 
      
  // (nt,nd,na,1,2,2)
  npJonesMatrices = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LJones, 3), PyArray_COMPLEX64, 0, 6);
  ptrJonesMatrices=p_complex64(npJonesMatrices);
  nt_Jones=(int)npJonesMatrices->dimensions[0];
  nd_Jones=(int)npJonesMatrices->dimensions[1];
  na_Jones=(int)npJonesMatrices->dimensions[2];
  nch_Jones=(int)npJonesMatrices->dimensions[3];
  JonesDims[0]=nt_Jones;
  JonesDims[1]=nd_Jones;
  JonesDims[2]=na_Jones;
  JonesDims[3]=nch_Jones;
  
  PyObject *_IDIR  = PyList_GetItem(LJones, 4);
  i_dir=(int) PyFloat_AsDouble(_IDIR);

  /* ////////////////////////////////////////////// */

  //i_dir=0;









  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX64, 0, 4);

  float complex* VisIn=p_complex64(NpVisIn);
  float complex* ThisVis;
  float complex VisCorr[4]={0};

  PyArrayObject *Np_l;
  Np_l = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LSM, 0), PyArray_FLOAT32, 0, 4);
  PyArrayObject *Np_m;
  Np_m = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LSM, 1), PyArray_FLOAT32, 0, 4);
  PyArrayObject *Np_I;
  Np_I = (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LSM, 2), PyArray_FLOAT32, 0, 4);
  
  PyArrayObject *NpWaveL;
  NpWaveL= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LFreqs, 0), PyArray_FLOAT32, 0, 4);
  PyArrayObject *NpFreqs;
  NpFreqs= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LFreqs, 1), PyArray_FLOAT32, 0, 4);
  PyArrayObject *NpDFreqs;
  NpDFreqs= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LFreqs, 2), PyArray_FLOAT32, 0, 4);
  float *p_DFreqs=p_float32(NpDFreqs);
  float *p_Freqs=p_float32(NpFreqs);

  PyArrayObject *NpUVW_dt;
  NpUVW_dt= (PyArrayObject *) PyArray_ContiguousFromObject(PyList_GetItem(LUVWSpeed, 0), PyArray_FLOAT32, 0, 4);
  //PyArrayObject *NpDFreqs;
  PyObject *_DT  = PyList_GetItem(LUVWSpeed, 1);
  float DT=(float) PyFloat_AsDouble(_DT);
  float *UVW_dt=p_float32(NpUVW_dt);

  PyObject *_FSmear  = PyList_GetItem(LSmearMode, 0);
  int FSmear=(int) PyFloat_AsDouble(_FSmear);
  PyObject *_TSmear  = PyList_GetItem(LSmearMode, 1);
  int TSmear=(int) PyFloat_AsDouble(_TSmear);





  UVWin=p_float64(NpUVWin);
  p_l=p_float32(Np_l);
  p_m=p_float32(Np_m);

  p_Flux=p_float32(Np_I);

  WaveL=p_float32(NpWaveL);
  
  int ch,dd,nchan,ndir;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];

  int ChanEquidistant=0;
  if(nchan>2){
    ChanEquidistant=1;
    float dFChan0=p_Freqs[1]-p_Freqs[0];
    for(ch=0; ch<(nchan-1); ch++){
      float df=abs(p_Freqs[ch+1]-p_Freqs[ch]);
      float ddf=abs(df-dFChan0);
      if(ddf>1){ChanEquidistant=0;}
    }
  }
  if(AllowChanEquidistant==0){
    ChanEquidistant=0;
  }


  ndir=Np_l->dimensions[0];

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

  float complex J0[4]={0},J1[4]={0},J0inv[4]={0},J1H[4]={0},J1Hinv[4]={0},JJ[4]={0};
  
  int ApplyJones=1;
  int irow;

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
    for ( irow=0; irow<nrow; irow++)  {
  	phase=(*UVWin++)*l;
  	//printf("cc %f \n",phase);
  	phase+=(*UVWin++)*m;
  	//printf("cc %f \n",phase);
  	phase+=(*UVWin++)*n;
  	//printf("cc %f \n",phase);


	if(ApplyJones==1){
	  int i_t=ptrTimeMappingJonesMatrices[irow];
	  int i_ant0=ptrA0[irow];
	  int i_ant1=ptrA1[irow];
	  //printf("%i %i %i %i | ",dd, i_t,i_ant0,i_ant1);
	  
	  GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant0, i_dir, J0);
	  GiveJones(ptrJonesMatrices, JonesDims, ptrCoefsInterp, i_t, i_ant1, i_dir, J1);
	  
	  MatH(J1,J1H);
	}



  	for(ch=0;ch<nchan;ch++){
	  ThisVis=VisIn+irow*nchan*4+ch*4;
	  /* VisCorr[0]=ThisVis[0]; */
	  /* VisCorr[1]=ThisVis[1]; */
	  /* VisCorr[2]=ThisVis[2]; */
	  /* VisCorr[3]=ThisVis[3]; */


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


  	  result=p_Flux[dd*nchan+ch]*Kernel;
  	  if(FSmear==1){
  	    //phi=PI*PI_C*p_DFreqs[ch]*phase;
  	    phi=PI*(p_DFreqs[ch]/C)*phase;
	    if(phi!=0.){
	      phi=sin(phi)/(phi);
	      result*=phi;
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
	      phi=sin(phi)/(phi);
	      result*=phi;
	    };
  	  };


  	  //printf("\n");

	  if(ApplyJones==1){
	    MatDot(J0,J1H,JJ);
	    ThisVis[0]   += JJ[0]*result;
	    ThisVis[1]   += JJ[1]*result;
	    ThisVis[2]   += JJ[2]*result;
	    ThisVis[3]   += JJ[3]*result;
	  }
	  else{
	    ThisVis[0]   += result;
	    ThisVis[3]   += result;
	  }
	  //printf("(%f,%f)\n",creal(ThisVis[0]),cimag(ThisVis[0]));


  	}
	
    }
  }

  //return Py_None;  
  return PyArray_Return(NpVisIn);
}


