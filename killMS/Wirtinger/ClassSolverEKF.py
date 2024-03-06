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
from killMS.Array.Dot import NpDotSSE
from killMS.Wirtinger.ClassJacobianAntenna import ClassJacobianAntenna
import killMS.Other.MyPickle

class ClassSolverEKF(ClassJacobianAntenna):
    def __init__(self, *args, **kwargs):
        ClassJacobianAntenna.__init__(self, *args, **kwargs)

    def GivePaPol(self,Pa_in,ipol):
        PaPol=Pa_in.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y,self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        PaPol=PaPol[:,ipol,:,:,ipol,:].reshape((self.NDir*self.NJacobBlocks_Y,self.NDir*self.NJacobBlocks_Y))
        return PaPol

    def setQxInvPol(self):
        QxInv=np.ones((self.NDir,1,self.NJacobBlocks_Y,self.NDir,1,self.NJacobBlocks_Y),np.float32)
        QxInv*=1./(self.AmpQx**2)
        QxInv=QxInv.reshape((self.NDir*self.NJacobBlocks_Y,self.NDir*self.NJacobBlocks_Y))
        self.LQxInv=[QxInv,QxInv]

    def PrepareJHJ_EKF(self,Pa_in,rms):
        self.L_JHJinv=[]
        incr=1
        # pylab.figure(1)
        # pylab.clf()
        # pylab.imshow(np.abs(self.JHJ),interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        for ipol in range(self.NJacobBlocks_X):
            PaPol=self.GivePaPol(Pa_in,ipol)
            Pinv=ModLinAlg.invSVD(PaPol)
            JHJ=self.L_JHJ[ipol]#*(1./rms**2)
            JHJ+=Pinv
            if self.DoReg:
                JHJ+=self.LQxInv[ipol]*(self.gamma**2)
            JHJinv=ModLinAlg.invSVD(JHJ)
            self.L_JHJinv.append(JHJinv)




    def CalcKapa_i(self,yr,Pa,rms):
        kapaout=0
        for ipol in range(self.NJacobBlocks_X):
            J=self.LJacob[ipol]
            PaPol=self.GivePaPol(Pa,ipol)
            pa=np.abs(np.diag(PaPol))
            pa=pa.reshape(1,pa.size)
            JP=J*pa
            trJPJH=np.sum(np.abs(JP*J.conj()))
            trYYH=np.sum(np.abs(yr)**2)
            Np=np.where(self.DicoData["flags_flat"]==0)[0].size
            Take=(self.DicoData["flags_flat"]==0)
            trR=np.sum(self.R_flat[Take])#Np*rms**2
            kapa=np.abs((trYYH-trR)/trJPJH)
            kapaout+=np.sqrt(kapa)
            #if self.iAnt==0:
            #    print("old",self.iAnt,rms,np.sqrt(kapa),trYYH,trR,trJPJH,pa)

        kapaout=np.max([1.,kapaout])
        return kapaout


    def CalcKapa_i_new(self,yr,Pa,rms):
        kapaout=0
        T=ClassTimeIt.ClassTimeIt("    Kapa")
        T.disable()
        iT=0
        for ipol in range(self.NJacobBlocks_X):
            J=self.LJacob[ipol]
            PaPol=self.GivePaPol(Pa,ipol)
            pa=np.abs(np.diag(PaPol))
            pa=pa.reshape(1,pa.size)
            T.timeit(iT); iT+=1
            nrow,_=J.shape
            flags=(self.DicoData["flags_flat"][ipol]==0)
            T.timeit(iT); iT+=1

            Weigths=self.Weights_flat[ipol].reshape((nrow,1))
            T.timeit(iT); iT+=1

            Jw=Weigths*J
            JP=Jw*pa
            T.timeit(iT); iT+=1

            trJPJH=np.sum(np.abs(JP[flags]*Jw[flags].conj()))
            T.timeit(iT); iT+=1

            YYH=np.abs(yr[ipol,flags])**2

            T.timeit(iT); iT+=1
            # Np=np.where(self.DicoData["flags_flat"]==0)[0].size
            # Take=(self.DicoData["flags_flat"]==0)
            
            T.timeit(iT); iT+=1
            R=(self.R_flat[ipol][flags])#Np*rms**2
            
            ww=(Weigths[flags]).ravel()
            
            trYYH_R=np.sum(ww**2*(YYH-R))
            T.timeit(iT); iT+=1
            kapa=np.abs(trYYH_R/trJPJH)
            #kapa=1
            kapaout+=np.sqrt(kapa)
            #if self.iAnt==0:
            #    print("new",self.iAnt,rms,np.sqrt(kapa),trYYH_R,trJPJH,pa)
        kapaout=np.max([1.,kapaout])

        return kapaout

    def ApplyK_vec(self,zr,rms,Pa,DoReg=True):

        

        # pylab.figure(1)
        # pylab.clf()
        # incr=1
        # pylab.subplot(2,1,1)
        # pylab.plot((z.flatten())[::incr].real)
        # pylab.plot((Jx.flatten())[::incr].real)
        # pylab.plot(zr.flatten()[::incr].real)
        # pylab.subplot(2,1,2)
        # pylab.plot((z.flatten())[::incr].imag)
        # pylab.plot((Jx.flatten())[::incr].imag)
        # pylab.plot(zr.flatten()[::incr].imag)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop


        # pylab.figure(1)
        # pylab.clf()
        # incr=1
        # pylab.subplot(2,1,1)
        # pylab.imshow(np.abs(self.JHJ))
        # pylab.subplot(2,1,2)
        # pylab.imshow(np.abs(self.JHJinv))
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop


        # if self.iAnt==5:
        #     pylab.figure(1)
        #     pylab.clf()
        #     incr=1
        #     f=(self.DicoData["flags_flat"]==0)
        #     pylab.subplot(2,1,1)
        #     pylab.plot((z1[f].flatten())[::incr].real)
        #     pylab.plot((zr[f].flatten())[::incr].real)
        #     pylab.plot((z1-zr)[f].flatten()[::incr].real)
        #     pylab.subplot(2,1,2)
        #     pylab.plot((z1[f].flatten())[::incr].imag)
        #     pylab.plot((zr[f].flatten())[::incr].imag)
        #     pylab.plot((z1-zr)[f].flatten()[::incr].imag)
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)

        Rinv_zr=self.Rinv_flat*zr
        JH_z=self.JH_z(Rinv_zr)

        x1 = self.JHJinv_x(JH_z)
        if (self.DoReg)&(DoReg):
            self.G0=np.ones_like(self.Ga)#*200.
            dx1a=self.Msq_x(self.LQxInv,(self.Ga-self.G0))
            dx1b = self.gamma*self.JHJinv_x(dx1a)
            print("x'_0:",dx1b)
            x1+=dx1b
            print("x':",x1)

        z1 = self.J_x(x1)

        zr-=z1
        zr*=self.Rinv_flat
        x2=self.JH_z(zr)

        if (self.DoReg)&(DoReg):
            xr=self.Ga.ravel()-self.G0.ravel()-self.gamma*x1.ravel()
            dx2=self.gamma*self.Msq_x(self.LQxInv,xr)
            x2+=dx2.reshape(x2.shape)

        x3=[]
        for ipol in range(self.NJacobBlocks_X):
            PaPol=self.GivePaPol(Pa,ipol)
            #print(PaPol,PaPol.shape)


            if self.TypeDot=="Numpy":
                Prod=np.dot(PaPol,x2[:,ipol,:].flatten())
            elif self.TypeDot=="SSE":
                X2=x2[:,ipol,:].flatten()
                X2=X2.reshape((1,X2.size))
                Prod=NpDotSSE.dot_A_BT(PaPol.copy(),X2)
            

            x3.append(Prod.reshape((self.NDir,1,self.NJacobBlocks_Y)))

            

        x3=np.concatenate(x3,axis=1)
        #x3=np.swapaxes(x3,1,2)

        return x3
    

        
    def doEKFStep(self,Gains,P,evP,rms,Gains0Iter=None):
        T=ClassTimeIt.ClassTimeIt("    EKF")
        T.disable()
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix(rms)
            self.SelectChannelKernelMat()
            T.timeit("CalcKernelMatrix")

        # print(self.iAnt,"MMMM",np.max(Gains),np.max(P),np.max(evP))
        # print(self.iAnt,"MMMM",np.max(Gains),np.max(P),np.max(evP))
        # print(self.iAnt,"MMMM",np.max(Gains),np.max(P),np.max(evP))

            
        z=self.DicoData["data_flat"]#self.GiveDataVec()

        f=(self.DicoData["flags_flat"]==0)
        ind=np.where(f)[0]
        Pa=P[self.iAnt]
        Ga=self.GiveSubVecGainAnt(Gains)
        self.Ga=Ga
        self.rms=rms

        #if self.iAnt==1:
        #    print(evP.ravel())
        self.rmsFromData=None
        if ind.size==0 or self.DataAllFlagged or self.ZeroSizedData:
            return Ga.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),Pa,{"std":-1.,"max":-1.,"kapa":-1.},0
        
        if self.DoReg:
            self.setQxInvPol()
        
        self.CalcJacobianAntenna(Gains)
        T.timeit("Jacob")
        
        # if Gains0Iter!=None:
        #     Ga=self.GiveSubVecGainAnt(Gains0Iter)

        Jx=self.J_x(Ga)
        T.timeit("J_x")

        

        self.PrepareJHJ_EKF(Pa,rms)
        T.timeit("PrepareJHJ")

        # estimate x
        zr=(z-Jx)
        zr[self.DicoData["flags_flat"]]=0
        
        T.timeit("Resid")
        
        kapa=self.CalcKapa_i(zr,Pa,rms)
        
        # try:
        #     kapa=self.CalcKapa_i_new(zr,Pa,rms)
        #     self.DicoData["Ga"]=Ga
        #     self.DicoData["zr"]=zr
        #     self.DicoData["rms"]=rms
        #     self.DicoData["KernelMat_AllChan"]=self.KernelMat_AllChan
        #     #killMS.Other.MyPickle.Save(self.DicoData,"DicoData_%i.pickle"%self.iAnt)
        #     #print(np.array([1])/0)
        # except:
        #     self.DicoData["Ga"]=Ga
        #     self.DicoData["zr"]=zr
        #     self.DicoData["rms"]=rms
        #     self.DicoData["Pa"]=Pa
        #     self.DicoData["KernelMat_AllChan"]=self.KernelMat_AllChan
        #     killMS.Other.MyPickle.Save(self.DicoData,"DicoData_crash_%i.pickle"%self.iAnt)
        #     stop

        # Weighted std estimate 
        zrs=zr[f]
        ws=np.absolute(self.DicoData["Rinv_flat"][f])
        std=np.sqrt(np.sum(ws*np.absolute(zrs)**2)/np.sum(ws))

        # # Original std estimate 
        # std=np.std(zr[f])


        InfoNoise={"std":std,"max":np.max(np.abs(zr[f])),"kapa":kapa}
        #print(self.iAnt,InfoNoise)
        #T.timeit("kapa")

        self.rmsFromData=np.std(zr[f])
        T.timeit("rmsFromData")

        # if np.isnan(self.rmsFromData):
        #     print(zr)
        #     print(zr[f])
        #     print(self.rmsFromData)
        #     stop

        # if self.iAnt==51:
        #     #self.DicoData["flags_flat"].fill(0)
        #     f=(self.DicoData["flags_flat"]==0)
        #     fig=pylab.figure(2)
        #     pylab.clf()
        #     pylab.plot((z[f]))#[::11])#[::11])
        #     pylab.plot((Jx[f]))#[::11])#[::11])

        #     #pylab.plot(zr[f])#[::11])#[::11])
        #     #pylab.draw()
        #     ifile=0
        #     while True:
        #         fname="png/png.%5.5i.png"%ifile
        #         if os.path.isfile(fname) :
        #             fig.savefig(fname)
        #             break
        #         ifile+=1
                
                
        #     #pylab.show(False)
        #     #pylab.pause(0.1)
        #     #stop


        x3=self.ApplyK_vec(zr,rms,Pa)
        
        T.timeit("ApplyK_vec")
        x0=Ga.flatten()
        x4=x0+self.LambdaKF*x3.flatten()

        # estimate P

        #Pa_new1=Pa-np.dot(evPa,Pa)
        evPa=evP[self.iAnt]
        Pa_new1=np.dot(evPa,Pa)
        #Pa_new1=Pa


        T.timeit("EstimateP")
        # ##################
        # for iPar in range(Pa.shape[0]):
        #     J_Px=self.J_x(Pa[iPar,:])
        #     xP=self.ApplyK_vec(J_Px,rms,Pa)
        #     evPa[iPar,:]=xP.flatten()
        # evPa= Pa-evPa
        # Pa_new1=evPa

        del(self.LJacob)
        T.timeit("Rest")
        
        # if self.iAnt==0:
        #     print(x4,Pa_new1,InfoNoise,evPa,Pa)

        # if np.max(np.abs(x4))>10:
        #     self.DicoData["Ga"]=Ga
        #     self.DicoData["Jx"]=Jx
        #     self.DicoData["zr"]=zr
        #     self.DicoData["rms"]=rms
        #     self.DicoData["Pa"]=Pa
        #     self.DicoData["x4"]=x4
        #     self.DicoData["ch0ch1"]=(self.ch0,self.ch1)
        #     self.DicoData["KernelMat_AllChan"]=self.KernelMat_AllChan
        #     killMS.Other.MyPickle.Save(self.DicoData,"DicoData_diverge_%i.pickle"%self.iAnt)
        #     stop

        return x4.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),Pa_new1,InfoNoise,1


    def CalcMatrixEvolveCov(self,Gains,P,rms):
        #print("EVOLVE!!!!!!!!!")
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix(rms)
            self.SelectChannelKernelMat()
        if self.LQxInv==None:
            self.setQxInvPol()
#            self.CalcKernelMatrix(rms)
        self.CalcJacobianAntenna(Gains)
        Pa=P[self.iAnt]
        NPars=Pa.shape[0]
        if self.DataAllFlagged:
            # print(self.iAnt,"OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            # print(self.iAnt,"OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            # print(self.iAnt,"OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            PaOnes=np.diag(np.ones((NPars,),self.CType))
            return PaOnes
        
        self.PrepareJHJ_EKF(Pa,rms)
        PaOnes=np.diag(np.ones((NPars,),self.CType))

        evPa=np.zeros_like(Pa)

        for iPar in range(Pa.shape[0]):
            J_Px=self.J_x(PaOnes[iPar,:])
            xP=self.ApplyK_vec(J_Px,rms,Pa,DoReg=False)
            evPa[iPar,:]=xP.flatten()


        evPa= PaOnes-evPa#(np.diag(np.diag(Pa-Pa_new)))#Pa-Pa_new#np.abs(np.diag(np.diag(Pa-Pa_new)))
        evPa=np.diag(np.diag(evPa))
        # print("CalcMatrixEvolveCov",self.iAnt,np.diag(evPa).max())
        # print("CalcMatrixEvolveCov",self.iAnt,np.diag(evPa).max())
        # print("CalcMatrixEvolveCov",self.iAnt,np.diag(evPa).max())
        # print("CalcMatrixEvolveCov",self.iAnt,np.diag(evPa).max())


        #print(evPa.min(),evPa.real.min())
        #print("=========Ev",self.iAnt,evPa)
        #if self.iAnt==0:
        #    print(evPa,Gains.ravel(),P.ravel())
        return evPa
