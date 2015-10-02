import numpy as np
import dotSSE

def dot_A_BT(A,BT):

    B=BT
    nxA,nyA=A.shape
    nxB,nyB=B.shape
    
    if nyA!=nyB: raise NameError("Matrices should have the same height [%i vs %i]"%(nyA,nyA))

    DType=A.dtype
    if DType==np.complex64:
        IntType=0
    if DType==np.complex128:
        IntType=1

    if not((A.dtype==np.complex64)|(A.dtype==np.complex128)): raise NameError("wrong dtype [%s]"%(str(A.dtype)))
    if not((B.dtype==np.complex64)|(B.dtype==np.complex128)): raise NameError("wrong dtype [%s]"%(str(B.dtype)))
    if not(A.dtype==B.dtype): raise NameError("A and B must have the same dtype [%s vs %s]"%(str(A.dtype),str(B.dtype)))

    C=np.zeros((nxA,nxB),DType)
    print "ok"
    dotSSE.dot(A,B,C,IntType)
    print "done"
    
    return C

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

    C=dot_A_BT(A,B.T.copy())

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
