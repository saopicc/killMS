
import numpy as np
import dotSSE
from killMS2.Other import ClassTimeIt
import NpDotSSE

def test():


    DType=np.complex128
    nx=10
    nyA=2
    A=DType(np.random.randn(nyA,nx)+1j*np.random.randn(nyA,nx))
    nyB=3
    B=DType(np.random.randn(nyB,nx)+1j*np.random.randn(nyB,nx))

    #A.fill(1)
    #B.fill(1)

    # A=A.T.copy()
    # B=B.T.copy()

    C=np.zeros((nyA,nyB),DType)

    print "==================================="
    print "A",A
    print "B",B
    if DType==np.complex64:
        IntType=0
    if DType==np.complex128:
        IntType=1

    print "=========="
    dotSSE.dot(A,B,C,IntType)

    print "=========="
    D=np.dot(A,B.T)
    print C-D

    A=np.ones((2000,100),np.complex64)
    B=A.copy()
    #C=np.zeros_like(A)

    T=ClassTimeIt.ClassTimeIt()
    for i in range(10):
        AA=np.dot(A.T,B)
        T.timeit("numpy")

    for i in range(10):

        #dotSSE.dot(A,B,C)
        NpDotSSE.dot_A_BT(A,B)
        T.timeit("sse")
    #print C

test()
