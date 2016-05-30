#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <time.h>
#include "complex.h"
#include <omp.h>



void dot_inner_64(const float complex * __restrict__ Avec, const float complex * __restrict__ Bvec, float complex * __restrict__ Cvec, size_t SIZE)
{

  const float complex * a = Avec;
  const float complex * b = Bvec;
  float complex *c = Cvec;
  size_t i;

  for (i = 0; i < SIZE; i++)
    {
      *c += a[i]*b[i];
    }

}


void dotSSE0_64(const float complex * __restrict__ A, const float complex * __restrict__ B, float complex * __restrict__ C, size_t sxA, size_t syA, size_t sxB, size_t syB)
{
  size_t ic,jc;
  

  {
    //#pragma omp parallel for
  for (ic = 0; ic < sxA; ic++)
    {
      
      const float complex * __restrict__ Avec=A+ic*syA;
      for (jc = 0; jc < sxB; jc++)
	{
	  const float complex * __restrict__ Bvec=B+jc*syB;
	  long off = ic*sxB+jc;
	  float complex * Cvec=C+off;
	  //printf("[ic,jc] = (%i, %i) [off=%i] (%f, %f)\n", (int)ic,(int)jc,off,creal(*Cvec),cimag(*Cvec));
	  dot_inner_64(Avec, Bvec, Cvec, syA);
	}
    }
  }

}

//==========================

void dot_inner_128(const double complex * __restrict__ Avec, const double complex * __restrict__ Bvec, double complex * __restrict__ Cvec, size_t SIZE)
{

  const double complex * a = Avec;
  const double complex * b = Bvec;
  double complex *c = Cvec;
  size_t i;

  for (i = 0; i < SIZE; i++)
    {
      *c += a[i]*b[i];
    }

}


void dotSSE0_128(const double complex * __restrict__ A, const double complex * __restrict__ B, double complex * __restrict__ C, size_t sxA, size_t syA, size_t sxB, size_t syB)
{
  size_t ic,jc;
  

  {
    //#pragma omp parallel for
  for (ic = 0; ic < sxA; ic++)
    {
      
      const double complex * __restrict__ Avec=A+ic*syA;
      for (jc = 0; jc < sxB; jc++)
	{
	  const double complex * __restrict__ Bvec=B+jc*syB;
	  long off = ic*sxB+jc;
	  double complex * Cvec=C+off;
	  //printf("[ic,jc] = (%i, %i) [off=%i] (%f, %f)\n", (int)ic,(int)jc,off,creal(*Cvec),cimag(*Cvec));
	  dot_inner_128(Avec, Bvec, Cvec, syA);
	}
    }
  }

}

