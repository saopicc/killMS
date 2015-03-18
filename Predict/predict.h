/* Header to test of C modules for arrays for Python: C_test.c */
#include "complex.h"

int *I_ptr(PyArrayObject *arrayin)  {
	return (int *) arrayin->data;
}


int *p_int32(PyArrayObject *arrayin)  {
  return (int *) arrayin->data;  /* pointer to arrayin data as double */
}


double *p_float64(PyArrayObject *arrayin)  {
  return (double *) arrayin->data;  /* pointer to arrayin data as double */
}

float *p_float32(PyArrayObject *arrayin)  {
  return (float *) arrayin->data;  /* pointer to arrayin data as double */
}


float complex *p_complex64(PyArrayObject *arrayin)  {
  return (float complex *) arrayin->data;  /* pointer to arrayin data as double */
}

double complex *p_complex128(PyArrayObject *arrayin)  {
  return (double complex *) arrayin->data;  /* pointer to arrayin data as double */
}


/* double *D_ptr(PyArrayObject *arrayin)  { */
/* 	return (double *) arrayin->data; */
/* } */

/* int *I_ptr(PyArrayObject *arrayin)  { */
/* 	return (int *) arrayin->data; */
/* } */


/* double complex *DC_ptr(PyArrayObject *arrayin)  { */
/* 	return (double complex *) arrayin->data; */
/* } */







static PyObject *predict(PyObject *self, PyObject *args);

