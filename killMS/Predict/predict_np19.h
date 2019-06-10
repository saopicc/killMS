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

/* Header to test of C modules for arrays for Python: C_test.c */
#define NPY_NO_DEPRECATED_API	NPY_1_8_API_VERSION
#include "complex.h"




static PyObject *predict(PyObject *self, PyObject *args);
static PyObject *predictJones(PyObject *self, PyObject *args);
static PyObject *CorrVis(PyObject *self, PyObject *args);
static PyObject *GiveMaxCorr(PyObject *self, PyObject *args);

/////////////////////////////////

void MatInv(float complex *A, float complex* B, int H ){
  float complex a,b,c,d,ff;

  if(H==0){
      a=A[0];
      b=A[1];
      c=A[2];
      d=A[3];}
  else{
    a=conj(A[0]);
    b=conj(A[2]);
    c=conj(A[1]);
    d=conj(A[3]);
  }  
  ff=1./((a*d-c*b));
  B[0]=ff*d;
  B[1]=-ff*b;
  B[2]=-ff*c;
  B[3]=ff*a;
}

void MatH(float complex *A, float complex* B){
  float complex a,b,c,d;

  a=conj(A[0]);
  b=conj(A[2]);
  c=conj(A[1]);
  d=conj(A[3]);
  B[0]=a;
  B[1]=b;
  B[2]=c;
  B[3]=d;
}

void MatDot(float complex *A, float complex* B, float complex* Out){
  float complex a0,b0,c0,d0;
  float complex a1,b1,c1,d1;

  a0=A[0];
  b0=A[1];
  c0=A[2];
  d0=A[3];
  
  a1=B[0];
  b1=B[1];
  c1=B[2];
  d1=B[3];
  
  Out[0]=a0*a1+b0*c1;
  Out[1]=a0*b1+b0*d1;
  Out[2]=c0*a1+d0*c1;
  Out[3]=c0*b1+d0*d1;

}
