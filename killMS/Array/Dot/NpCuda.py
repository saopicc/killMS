#!/usr/bin/env python
"""
killMS, a package for calibration in radio interferometry.
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import scikits.cuda.linalg as culinalg
culinalg.init()
from killMS.Other import ClassTimeIt
import time
import numpy as np
import cudamat as cm
cm.cublas_init()


def Test():


    A=np.float32(np.random.randn(*(2000,2000)))
    A=np.complex64(np.ones((2000,2000))+1j*np.ones((2000,2000)))
    AT=A.T.copy()

    A_32=A#np.float32(A)
    AT_32=AT#np.float32(AT)


    T=ClassTimeIt.ClassTimeIt()
    # create two random matrices and copy them to the GPU
    g_A0 = cm.CUDAMatrix(A)
    g_AT0 = cm.CUDAMatrix(AT)

    
    # perform calculations on the GPU
    P0 = cm.dot(g_AT0, g_A0).asarray()
    #d = cm.sum(axis = 0)
    T.timeit("GPU0")
    del(g_AT0,g_A0)
    #T.reinit()

    # copy d back to the host (CPU) and print


    g_A1 = gpuarray.to_gpu(A)
    g_AT1 = gpuarray.to_gpu(AT)
    #time.sleep(5)
    
    #T.timeit("tranf0")
    g_P1 = culinalg.dot(g_AT1,g_A1)

    P1=g_P1.get()

    #T.timeit("tranf1")
    T.timeit("GPU1")
    
    np_P=np.dot(AT,A)
    T.timeit("np")
    #print g_P-np_P


    print(np.max(np_P-P0))
    print(np.max(np_P-P1))


