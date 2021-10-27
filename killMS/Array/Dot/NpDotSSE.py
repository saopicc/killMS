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
import numpy as np
try:
    from . import dotSSE
except ImportError:
    from killMS.cbuild.Array.Dot import dotSSE

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
    dotSSE.dot(A,B,C,IntType)
    
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

    print("===================================")
    print("A",A)
    print("B",B)
    
    print("==========")
    dotSSE.dot(A,B,C)
    print(C)

    print("==========")
    print(np.dot(A,B.T))


    # T=ClassTimeIt.ClassTimeIt()
    # for i in range(10):
    #     AA=np.dot(A.T,B)
    #     T.timeit("numpy")
    # for i in range(10):
    #     dotSSE.dot(A,B,C)
    #     T.timeit("sse")
    # #print C
