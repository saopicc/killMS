#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <time.h>
#include "complex.h"
#include <omp.h>

//#include <stdlib.h>
//#include <math.h>

//#define SIZE 65535L //(1L << 16)



void dot_inner(const float complex * __restrict__ Avec, const float complex * __restrict__ Bvec, float complex * __restrict__ Cvec, size_t SIZE)
//void dot_inner(const float * __restrict__ Avec, const float * __restrict__ Bvec, float * __restrict__ Cvec, size_t SIZE)
{

  /* const float complex * __restrict__ a = __builtin_assume_aligned(Avec, 8); */
  /* const float complex * __restrict__ b = __builtin_assume_aligned(Bvec, 8); */
  /* float complex *c = __builtin_assume_aligned(Cvec, 8); */

  const float complex * a = Avec;
  const float complex * b = Bvec;
  float complex *c = Cvec;
  size_t i;

  for (i = 0; i < SIZE; i++)
    {
      //printf("  [%i] a (%f, %f)\n",(int)i,creal(a[i]),cimag(a[i]));
      //printf("  [%i] b (%f, %f)\n",(int)i,creal(b[i]),cimag(b[i]));
      *c += a[i]*b[i];
    }

}

/* void dot_inner(const float complex * __restrict__ Avec, const float complex * __restrict__ Bvec, float complex * __restrict__ Cvec, size_t SIZE) */
/* //void dot_inner(const float * __restrict__ Avec, const float * __restrict__ Bvec, float * __restrict__ Cvec, size_t SIZE) */
/* { */

/*   /\* const float complex * __restrict__ a = __builtin_assume_aligned(Avec, 8); *\/ */
/*   /\* const float complex * __restrict__ b = __builtin_assume_aligned(Bvec, 8); *\/ */
/*   /\* float complex *c = __builtin_assume_aligned(Cvec, 8); *\/ */

/*   const float complex * a = Avec; */
/*   const float complex * b = Bvec; */
/*   float complex *c = Cvec; */
/*   size_t i; */

/*   /\* int alig=8; *\/ */
/*   /\* const float complex *a = __builtin_assume_aligned(Avec, alig); *\/ */
/*   /\* const float complex *b = __builtin_assume_aligned(Bvec, alig); *\/ */
/*   /\* float complex *c = __builtin_assume_aligned(Cvec, alig); *\/ */
/*   /\* size_t i; *\/ */

/*   for (i = 0; i < SIZE; i++) */
/*     { */
/*       //printf("  [%i] a (%f, %f)\n",(int)i,creal(a[i]),cimag(a[i])); */
/*       //printf("  [%i] b (%f, %f)\n",(int)i,creal(b[i]),cimag(b[i])); */
/*       *c += a[i]*b[i]; */
/*     } */

/* } */

void dotSSE0(const float complex * __restrict__ A, const float complex * __restrict__ B, float complex * __restrict__ C, size_t sxA, size_t syA, size_t sxB, size_t syB)
{
  size_t ic,jc;
  

  {
#pragma omp parallel for
  for (ic = 0; ic < sxA; ic++)
    {
      
      const float complex * __restrict__ Avec=A+ic*syA;
      for (jc = 0; jc < sxB; jc++)
	{
	  const float complex * __restrict__ Bvec=B+jc*syB;
	  long off = ic*sxB+jc;
	  float complex * Cvec=C+off;
	  //printf("[ic,jc] = (%i, %i) [off=%i] (%f, %f)\n", (int)ic,(int)jc,off,creal(*Cvec),cimag(*Cvec));
	  dot_inner(Avec, Bvec, Cvec, syA);
	}
    }
  }

}


/* long SIZE=165535; */

/* void test4(double * __restrict__ a, double * __restrict__ b) */
/* { */
/* 	int i; */

/* 	double *x = __builtin_assume_aligned(a, 16); */
/* 	double *y = __builtin_assume_aligned(b, 16); */

/* 	for (i = 0; i < SIZE; i++) */
/* 	{ */
/* 		x[i] += y[i]; */
/* 	} */
/* } */
