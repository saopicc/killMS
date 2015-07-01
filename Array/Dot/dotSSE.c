/* A file to test imorting C modules for handling arrays to Python */
//#define NPY_NO_DEPRECATED_API	NPY_1_8_API_VERSION

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include "complex.h"
#include "dotSSE.h"
#include <assert.h>
#include <stdio.h>
#include "dot0.c"

/* #### Globals #################################### */

/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */

/* ==== Set up the methods table ====================== */
static PyMethodDef dotSSE_Methods[] = {
	{"dot", dot0, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void initdotSSE()  {
	(void) Py_InitModule("dotSSE", dotSSE_Methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}



static PyObject *dot0(PyObject *self, PyObject *args)
{
  PyArrayObject *NpA,*NpB,*NpC;
  
  if (!PyArg_ParseTuple(args, "O!O!O!",
			&PyArray_Type, &NpA,
			&PyArray_Type, &NpB,
			&PyArray_Type, &NpC))  return NULL;

  float complex *A,*B,*C;

  A=p_complex64(NpA);
  B=p_complex64(NpB);
  C=p_complex64(NpC);

  size_t sxA, syA, sxB, syB;
  sxA=NpA->dimensions[0];
  syA=NpA->dimensions[1];
  sxB=NpB->dimensions[0];
  syB=NpB->dimensions[1];
  
  //printf("shape A = [%i, %i]\n",(int)(NpA->dimensions[0]),(int)(NpA->dimensions[1]));
  //printf("shape B = [%i, %i]\n",(int)(NpB->dimensions[0]),(int)(NpB->dimensions[1]));

  dotSSE0(A, B, C, sxA, syA, sxB, syB);

  Py_INCREF(Py_None);
  return Py_None;
}

