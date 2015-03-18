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
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void initpredict()  {
	(void) Py_InitModule("predict", predict_Methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}


static PyObject *predict(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyObject *LSM, *LUVWSpeed, *LFreqs,*LSmearMode;
  PyArrayObject *NpVisIn, *NpUVWin, *matout;
  float *p_l,*p_m,*p_alpha,*p_Flux, *WaveL;

  double *UVWin;
  int nrow,npol,nsources,i,dim[2];
  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!",
			&ObjVisIn,
			&PyArray_Type, &NpUVWin, 
			&PyList_Type, &LFreqs,
			&PyList_Type, &LSM,
			&PyList_Type, &LUVWSpeed,
			&PyList_Type, &LSmearMode))  return NULL;
  
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
  	  //printf("ch: %i %f\n",ch,WaveL[ch]);
  	  result=p_Flux[dd*nchan+ch]*cexp(phase*c1[ch]);
  	  if(FSmear==1){
  	    phi=PI*PI_C*p_DFreqs[ch]*phase;
  	    phi=sin(phi)/(phi);
  	    result*=phi;
  	  };
  	  if(TSmear==1){
  	    du=UVW_dt[3*i]*l;
  	    dv=UVW_dt[3*i+1]*m;
  	    dw=UVW_dt[3*i+2]*n;
  	    dphase=(du+dv+dw)*DT;
  	    phi=PI*PI_C*p_Freqs[ch]*dphase;
  	    //printf("phi = %f\n",phi);
  	    //printf("dphase = %f\n",dphase);
  	    phi=sin(phi)/(phi);
  	    result*=phi;
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


