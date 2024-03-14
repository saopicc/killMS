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

static struct PyModuleDef dotSSE =
{
    PyModuleDef_HEAD_INIT,
    "dotSSE",    /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    dotSSE_Methods
};


/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
PyMODINIT_FUNC PyInit_dotSSE(void)
{
    PyObject * m = PyModule_Create(&dotSSE);
    import_array();
    return m;
}



static PyObject *dot0(PyObject *self, PyObject *args)
{
  PyArrayObject *NpA,*NpB,*NpC;
  int IntType;
  if (!PyArg_ParseTuple(args, "O!O!O!i",
			&PyArray_Type, &NpA,
			&PyArray_Type, &NpB,
			&PyArray_Type, &NpC,
			&IntType))  return NULL;

  if(IntType==0){
    float complex *A,*B,*C;
    
    A=p_complex64(NpA);
    B=p_complex64(NpB);
    C=p_complex64(NpC);
    
    size_t sxA, syA, sxB, syB;
    sxA=NpA->dimensions[0];
    syA=NpA->dimensions[1];
    sxB=NpB->dimensions[0];
    syB=NpB->dimensions[1];
    
    dotSSE0_64(A, B, C, sxA, syA, sxB, syB);
  }
  if(IntType==1){
    double complex *A,*B,*C;
    
    A=p_complex128(NpA);
    B=p_complex128(NpB);
    C=p_complex128(NpC);
    
    size_t sxA, syA, sxB, syB;
    sxA=NpA->dimensions[0];
    syA=NpA->dimensions[1];
    sxB=NpB->dimensions[0];
    syB=NpB->dimensions[1];
    
    dotSSE0_128(A, B, C, sxA, syA, sxB, syB);
  }
  
  Py_INCREF(Py_None);
  return Py_None;
}

