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
from killMS.Array import NpShared
from killMS.Predict.PredictGaussPoints_NumExpr5 import ClassPredict
import os
from killMS.Data import ClassVisServer
#from Sky import ClassSM
from killMS.Array import ModLinAlg
from killMS.Other import ClassTimeIt
from killMS.Wirtinger.ClassJacobianAntenna import ClassJacobianAntenna

class ClassSolverLM(ClassJacobianAntenna):
    def __init__(self, *args, **kwargs):
        ClassJacobianAntenna.__init__(self, *args, **kwargs)

    def PrepareJHJ_LM(self):

        self.L_JHJinv=[]
        if self.DataAllFlagged:
            return

        for ipol in range(self.NJacobBlocks_X):

            M=self.L_JHJ[ipol]
            if self.DoTikhonov:
                self.LambdaTkNorm=self.LambdaTk*np.mean(np.abs(np.diag(M)))
                
                # Lin.shape= (self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)
                Linv=np.diag(self.Linv[:,ipol,:].ravel())
                Linv*=self.LambdaTkNorm/(1.+self.LambdaLM)
                M2=M+Linv

                # pylab.clf()
                # pylab.subplot(1,2,1)
                # pylab.imshow(np.abs(M),interpolation="nearest")
                # pylab.colorbar()
                # pylab.subplot(1,2,2)
                # pylab.imshow(np.abs(Linv),interpolation="nearest")
                # pylab.colorbar()
                # pylab.draw()
                # pylab.show(False)
                # pylab.pause(0.1)
                # stop

            else:
                M2=M


            JHJinv=ModLinAlg.invSVD(M2)
            #JHJinv=ModLinAlg.invSVD(self.JHJ)
            self.L_JHJinv.append(JHJinv)

    def doLMStep(self,Gains):
        # if self.iAnt==55:
        #     print(self.iAnt,Gains)
        T=ClassTimeIt.ClassTimeIt("doLMStep")
        T.disable()

        #Gains.fill(1.)
        
#         A=np.random.randn(10000,100)+1j*np.random.randn(10000,100)
#         B=np.random.randn(10000,100)+1j*np.random.randn(10000,100)
#         AT=A.T#.conj().copy()
# #        AT=A.T
#         # A=np.require(A,requirements='F_CONTIGUOUS')
#         # AT=np.require(AT,requirements='F_CONTIGUOUS')
#         # A=np.require(A,requirements='F')
#         # AT=np.require(AT,requirements='F')


#         T=ClassTimeIt.ClassTimeIt("doLMStep")
#         for i in range(20):
#             np.dot(AT,B)

#         T.timeit("%i"%i)

        
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix()
            self.SelectChannelKernelMat()
            T.timeit("CalcKernelMatrix")

        Ga=self.GiveSubVecGainAnt(Gains)


        if self.DoCompress:
            flags_key="flags_flat_avg"
            data_key="data_flat_avg"
            if self.DoMergeStations:
                flags_key="flags_flat_avg_merged"
                data_key="data_flat_avg_merged"
        else:
            flags_key="flags_flat"
            data_key="data_flat"
            
        f=(self.DicoData[flags_key]==0)
        
        # ind=np.where(f)[0]
        # if self.iAnt==56:
        #     print ind.size/float(f.size),np.abs(Gains[self.iAnt,0,0,0])

        
        if self.DataAllFlagged:
            return Ga.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),None,{"std":-1.,"max":-1.,"kapa":None}



        # if ind.size==0:
        #     return Ga.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),None,{"std":-1.,"max":-1.,"kapa":None}


        z=self.DicoData[data_key]#self.GiveDataVec()
        

        self.CalcJacobianAntenna(Gains)
        T.timeit("CalcJacobianAntenna")
        self.PrepareJHJ_LM()
        T.timeit("PrepareJHJ_L")



        T.timeit("GiveSubVecGainAnt")
        Jx=self.J_x(Ga)
        T.timeit("Jx")
        zr=z-Jx

        zr[self.DicoData[flags_key]]=0
        T.timeit("resid")

        # JH_z_0=np.load("LM.npz")["JH_z"]
        # x1_0=np.load("LM.npz")["x1"]
        # z_0=np.load("LM.npz")["z"]
        # Jx_0=np.load("LM.npz")["Jx"]




        InfoNoise={"std":np.std(zr[f]),"max":np.max(np.abs(zr[f])),"kapa":None}


        JH_z=self.JH_z(zr)
        T.timeit("JH_z")
        #self.JHJinv=ModLinAlg.invSVD(self.JHJ)
        #self.JHJinv=np.linalg.inv(self.JHJ)
        xi=Ga.flatten()
        T.timeit("self.JHJinv_x")
        

        if self.DoTikhonov:
            self.LambdaTkNorm
            Gi=xi.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
            JH_z=JH_z.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
            for polIndex in range(self.NJacobBlocks_X):
                gireg=Gi[:,polIndex,:]
                #gi=JH_z[:,polIndex,:]
                x0reg=self.X0[:,polIndex,:]
                Linv=(self.Linv[:,polIndex,:])
                JH_z[:,polIndex,:]-=self.LambdaTkNorm*Linv*(gireg-x0reg)
        dx = (1./(1.+self.LambdaLM)) * self.JHJinv_x(JH_z)

        
        
        # #print self.iAnt
        # if True:#self.iAnt==55:
        #     f=(self.DicoData[flags_key]==0)
        #     import pylab
        #     fig=pylab.figure(1)
        #     op0=np.abs
        #     op1=np.real
        #     pylab.clf()
        #     pylab.subplot(1,3,1)
        #     #pylab.plot(op0(z[f])[::1]**2)#[::11])
        #     pylab.plot( op1( z[f])[::1] )#[::11])
        #     #pylab.ylim(0,800)
        #     pylab.subplot(1,3,2)
        #     #pylab.plot(op0(Jx[f])[::1]**2)#[::11])
        #     pylab.plot( op1(Jx[f])[::1] )#[::11])
        #     #pylab.ylim(0,800)
        #     pylab.subplot(1,3,3)
        #     #pylab.plot(op0(zr[f])[::1])#[::11])
        #     pylab.plot( op1(zr[f])[::1] )#[::11])
        #     #pylab.ylim(-30,30)
        #     pylab.draw()
        #     pylab.show(block=False)
        #     pylab.pause(0.1)
            
        #     # iF=0
        #     # while True:
        #     #     fName="Graph_%i_%i.png"%(self.iAnt,iF)
        #     #     import os
        #     #     if not os.path.isfile(fName):
        #     #         break
        #     #     else:
        #     #         iF+=1
        #     # fig.savefig(fName)
            


        # # pylab.figure(2)
        # # pylab.clf()
        # # #pylab.plot((z)[::11])
        # # #pylab.plot((Jx-Jx_0)[::11])
        # # #pylab.plot(zr[::11])
        # # #pylab.plot(JH_z.flatten())
        # # #pylab.plot(JH_z_0.flatten())
        # # pylab.plot(x1.flatten())
        # # pylab.plot(x1_0.flatten())
        # # pylab.draw()
        # # pylab.show(False)
        # # pylab.pause(0.1)

        # stop
        # # np.savez("LM",JH_z=JH_z,x1=x1,z=z,Jx=Jx)
 
        # print JH_z.shape

        dx+=xi
        del(self.LJacob)
        T.timeit("rest")
        # print self.iAnt,np.mean(x1),x1.size,ind.size
        
        xout=dx.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        # print self.iAnt,xout.ravel()
        return xout,None,InfoNoise
