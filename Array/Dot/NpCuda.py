import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import scikits.cuda.linalg as culinalg
culinalg.init()
from killMS2.Other import ClassTimeIt


def Test():


    A=np.float64(np.random.randn(*(50000,100)))
    AT=A.T.copy()

    A_32=np.float32(A)
    AT_32=np.float32(AT)

    T=ClassTimeIt.ClassTimeIt()
    g_A = gpuarray.to_gpu(A_32)
    g_AT = gpuarray.to_gpu(AT_32)
    g_prod = culinalg.dot(g_AT,g_A)
    g_P=g_prod.get()
    T.timeit("GPU")
    
    np_P=np.dot(A.T,A)
    T.timeit("np")
    print g_P-np_P
