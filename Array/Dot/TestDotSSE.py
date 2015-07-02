
import numpy as np
import dotSSE
#import ClassTimeIt

def test():

    nx=10
    nyA=2
    A=np.complex64(np.random.randn(nyA,nx)+1j*np.random.randn(nyA,nx))
    nyB=3
    B=np.complex64(np.random.randn(nyB,nx)+1j*np.random.randn(nyB,nx))

    #A.fill(1)
    #B.fill(1)

    # A=A.T.copy()
    # B=B.T.copy()

    C=np.zeros((nyA,nyB),np.complex64)

    print "==================================="
    print "A",A
    print "B",B
    
    print "=========="
    dotSSE.dot(A,B,C)
    print C

    print "=========="
    print np.dot(A,B.T)


    # T=ClassTimeIt.ClassTimeIt()
    # for i in range(10):
    #     AA=np.dot(A.T,B)
    #     T.timeit("numpy")
    # for i in range(10):
    #     dotSSE.dot(A,B,C)
    #     T.timeit("sse")
    # #print C

test()
