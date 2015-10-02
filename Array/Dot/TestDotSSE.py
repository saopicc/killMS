
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

    #A=np.complex64(np.random.rand(2000,100)+1j*np.random.rand(2000,100))
    A=np.complex128( np.ones((10000,100)))
    B=A.copy()
    #C=np.zeros_like(A)


    N=1
    T=ClassTimeIt.ClassTimeIt()
    for i in range(N):
        AA=np.dot(A.T,B)
        print AA.shape
        T.timeit("numpy")

    A=A.T.copy()
    B=B.T.copy()
    T=ClassTimeIt.ClassTimeIt()
    for i in range(N):

        #dotSSE.dot(A,B,C)
        C=NpDotSSE.dot_A_BT(A,B)
        print C.shape
        T.timeit("sse")
    #print C

test()
