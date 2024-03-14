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
from killMS.Other import ClassTimeIt
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

    print("===================================")
    print("A",A)
    print("B",B)
    if DType==np.complex64:
        IntType=0
    if DType==np.complex128:
        IntType=1

    print("==========")
    dotSSE.dot(A,B,C,IntType)

    print("==========")
    D=np.dot(A,B.T)
    print(C-D)

    #A=np.complex64(np.random.rand(2000,100)+1j*np.random.rand(2000,100))
    A=np.complex64( np.ones((50000,100)))
    B=A.copy()
    #C=np.zeros_like(A)


    N=10
    T=ClassTimeIt.ClassTimeIt()
    for i in range(N):
        AA=np.dot(A.T,B)
        print(AA.shape)
        T.timeit("numpy")

    A=A.T.copy()
    B=B.T.copy()
    T=ClassTimeIt.ClassTimeIt()
    for i in range(N):

        #dotSSE.dot(A,B,C)
        C=NpDotSSE.dot_A_BT(A,B)
        print(C.shape)
        T.timeit("sse")
    #print C

test()
