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
from pyrap.tables import table
from killMS.Data.ClassMS import ClassMS
from SkyModel.Sky.ClassSM import ClassSM
from killMS.Other.ClassTimeIt import ClassTimeIt
import numexpr as ne
#import ModNumExpr
#from DDFacet.Other.progressbar import ProgressBar
import multiprocessing
from killMS.Array import ModLinAlg

#ne.evaluate=lambda sin: ("return %s"%sin)


class ClassPredict():
    def __init__(self,Precision="D",NCPU=6):
        self.NCPU=NCPU
        ne.set_num_threads(self.NCPU)
        if Precision=="D":
            self.CType=np.complex128
            self.FType=np.float64
        if Precision=="S":
            self.CType=np.complex64
            self.FType=np.float32

    def ApplyCal(self,DicoData,ApplyTimeJones,iCluster):
        D=ApplyTimeJones
        Beam=D["Beam"]
        BeamH=D["BeamH"]
        lt0,lt1=D["t0"],D["t1"]
        ColOutDir=DicoData["data"]
        A0=DicoData["A0"]
        A1=DicoData["A1"]
        times=DicoData["times"]
        na=int(DicoData["infos"][0])

        # nt,nd,nd,nchan,_,_=Beam.shape
        # med=np.median(np.abs(Beam))
        # Threshold=med*1e-2
        
        for it in range(lt0.size):
            t0,t1=lt0[it],lt1[it]
            ind=np.where((times>=t0)&(times<t1))[0]
            if ind.size==0: continue
            data=ColOutDir[ind]
            # flags=DicoData["flags"][ind]
            A0sel=A0[ind]
            A1sel=A1[ind]
            #print("CACA",ChanMap)
            if "ChanMap" in ApplyTimeJones.keys():
                ChanMap=ApplyTimeJones["ChanMap"]
            else:
                ChanMap=range(nf)

            for ichan in range(len(ChanMap)):
                JChan=ChanMap[ichan]
                if iCluster!=-1:
                    J0=Beam[it,iCluster,:,JChan,:,:].reshape((na,4))
                    JH0=BeamH[it,iCluster,:,JChan,:,:].reshape((na,4))
                else:
                    J0=np.mean(Beam[it,:,:,JChan,:,:],axis=1).reshape((na,4))
                    JH0=np.mean(BeamH[it,:,:,JChan,:,:],axis=1).reshape((na,4))
                
                J=ModLinAlg.BatchInverse(J0)
                JH=ModLinAlg.BatchInverse(JH0)

                data[:,ichan,:]=ModLinAlg.BatchDot(J[A0sel,:],data[:,ichan,:])
                data[:,ichan,:]=ModLinAlg.BatchDot(data[:,ichan,:],JH[A1sel,:])

                # Abs_g0=(np.abs(J0[A0sel,0])<Threshold)
                # Abs_g1=(np.abs(JH0[A1sel,0])<Threshold)
                # flags[Abs_g0,ichan,:]=True
                # flags[Abs_g1,ichan,:]=True

            ColOutDir[ind]=data[:]
            # DicoData["flags"][ind]=flags[:]

    def predictKernelPolCluster(self,DicoData,SM,iDirection=None,ApplyJones=None,ApplyTimeJones=None,Noise=None,VariableFunc=None):
        T=ClassTimeIt("predictKernelPolCluster")
        T.disable()
        self.DicoData=DicoData
        self.SourceCat=SM.SourceCat

        freq=DicoData["freqs"]
        times=DicoData["times"]
        nf=freq.size
        na=int(DicoData["infos"][0])
        
        nrows=DicoData["A0"].size
        DataOut=np.zeros((nrows,nf,4),self.CType)
        if nrows==0: return DataOut
        
        self.freqs=freq
        self.wave=299792458./self.freqs
        
        if iDirection!=None:
            ListDirection=[iDirection]
        else:
            ListDirection=SM.Dirs#range(SM.NDir)
        T.timeit("0")
        A0=DicoData["A0"]
        A1=DicoData["A1"]
        if ApplyJones!=None:
            na,NDir,_=ApplyJones.shape
            Jones=np.swapaxes(ApplyJones,0,1)
            Jones=Jones.reshape((NDir,na,4))
            JonesH=ModLinAlg.BatchH(Jones)
        T.timeit("1")


        for iCluster in ListDirection:
            print("IIIIIIIIIIIIIIIIII",iCluster)
            ColOutDir=self.PredictDirSPW(iCluster)
            T.timeit("2")
            if ColOutDir is None: continue


            # print(iCluster,ListDirection)
            # print(ColOutDir.shape)
            # ColOutDir.fill(0)
            # print(ColOutDir.shape)
            # ColOutDir[:,:,0]=1
            # print(ColOutDir.shape)
            # ColOutDir[:,:,3]=1
            # print(ColOutDir.shape)

            # Apply Jones
            if ApplyJones!=None:

                J=Jones[iCluster]
                JH=JonesH[iCluster]
                for ichan in range(nf):
                    ColOutDir[:,ichan,:]=ModLinAlg.BatchDot(J[A0,:],ColOutDir[:,ichan,:])
                    ColOutDir[:,ichan,:]=ModLinAlg.BatchDot(ColOutDir[:,ichan,:],JH[A1,:])
            T.timeit("3")

            if VariableFunc is not None:#"DicoBeam" in DicoData.keys():
                tt=np.unique(times)
                lt0,lt1=tt[0:-1],tt[1::]
                for it in range(lt0.size):
                    t0,t1=lt0[it],lt1[it]
                    ind=np.where((times>=t0)&(times<t1))[0]
                    if ind.size==0: continue
                    data=ColOutDir[ind]

                    if "ChanMap" in ApplyTimeJones.keys():
                        ChanMap=ApplyTimeJones["ChanMap"]
                    else:
                        ChanMap=range(nf)

                    for ichan in range(len(ChanMap)):
                        tc=(t0+t1)/2.
                        nuc=freq[ichan]
                        ColOutDir[ind,ichan,:]*=VariableFunc(tc,nuc)
                        # c0=ColOutDir[ind,ichan,:].copy()
                        # ColOutDir[ind,ichan,:]*=VariableFunc(tc,nuc)
                        # print(c0-ColOutDir[ind,ichan,:])
                        #print(it,ichan,VariableFunc(tc,nuc))

            if ApplyTimeJones is not None:#"DicoBeam" in DicoData.keys():
                D=ApplyTimeJones#DicoData["DicoBeam"]
                Beam=D["Beam"]
                BeamH=D["BeamH"]

                lt0,lt1=D["t0"],D["t1"]


                for it in range(lt0.size):
                    t0,t1=lt0[it],lt1[it]
                    ind=np.where((times>=t0)&(times<t1))[0]
                    if ind.size==0: continue
                    data=ColOutDir[ind]
                    A0sel=A0[ind]
                    A1sel=A1[ind]

                    if "ChanMap" in ApplyTimeJones.keys():
                        ChanMap=ApplyTimeJones["ChanMap"]
                    else:
                        ChanMap=range(nf)

                    #print("ChanMap:",ChanMap)

                    for ichan in range(len(ChanMap)):
                        JChan=ChanMap[ichan]
                        J=Beam[it,iCluster,:,JChan,:,:].reshape((na,4))
                        JH=BeamH[it,iCluster,:,JChan,:,:].reshape((na,4))
                        data[:,ichan,:]=ModLinAlg.BatchDot(J[A0sel,:],data[:,ichan,:])
                        data[:,ichan,:]=ModLinAlg.BatchDot(data[:,ichan,:],JH[A1sel,:])


                    ColOutDir[ind]=data[:]
            T.timeit("4")
            

            DataOut+=ColOutDir
            T.timeit("5")


        if Noise is not None:
            DataOut+=Noise/np.sqrt(2.)*(np.random.randn(*ColOutDir.shape)+1j*np.random.randn(*ColOutDir.shape))

        return DataOut


    def PredictDirSPW(self,idir):
        T=ClassTimeIt("PredictDirSPW")
        T.disable()
        ind0=np.where(self.SourceCat.Cluster==idir)[0]
        NSource=ind0.size
        if NSource==0: return None
        SourceCat=self.SourceCat[ind0]
        freq=self.freqs
        pi=np.pi
        wave=self.wave#[0]

        uvw=self.DicoData["uvw"]

        U=self.FType(uvw[:,0].flatten().copy())
        V=self.FType(uvw[:,1].flatten().copy())
        W=self.FType(uvw[:,2].flatten().copy())

        U=U.reshape((1,U.size,1,1))
        V=V.reshape((1,U.size,1,1))
        W=W.reshape((1,U.size,1,1))
        T.timeit("0")
        
        #ColOut=np.zeros(U.shape,dtype=complex)
        f0=self.CType(2*pi*1j/wave)
        f0=f0.reshape((1,1,f0.size,1))

        rasel =SourceCat.ra
        decsel=SourceCat.dec
        
        TypeSources=SourceCat.Type
        Gmaj=SourceCat.Gmaj.reshape((NSource,1,1,1))
        Gmin=SourceCat.Gmin.reshape((NSource,1,1,1))
        Gangle=SourceCat.Gangle.reshape((NSource,1,1,1))

        # print("%i:"%idir)
        # print(Gmin,Gmaj,Gangle)
        # print()
        RefFreq=SourceCat.RefFreq.reshape((NSource,1,1,1))
        alpha=SourceCat.alpha.reshape((NSource,1,1,1))

        fI=SourceCat.I.reshape((NSource,1,1))
        fQ=SourceCat.Q.reshape((NSource,1,1))
        fU=SourceCat.U.reshape((NSource,1,1))
        fV=SourceCat.V.reshape((NSource,1,1))
        Sky=np.zeros((NSource,1,1,4),self.CType)
        Sky[:,:,:,0]=(fI+fQ);
        Sky[:,:,:,1]=(fU+1j*fV);
        Sky[:,:,:,2]=(fU-1j*fV);
        Sky[:,:,:,3]=(fI-fQ);

        Ssel  =Sky*(freq.reshape((1,1,freq.size,1))/RefFreq)**(alpha)
        Ssel=self.CType(Ssel)
        T.timeit("1")




        Ll=self.FType(SourceCat.l)
        Lm=self.FType(SourceCat.m)
        
        #print(Ssel,Ll, Lm)

        l=Ll.reshape(NSource,1,1,1)
        m=Lm.reshape(NSource,1,1,1)
        nn=self.FType(np.sqrt(1.-l**2-m**2)-1.)
        f=Ssel
        Ssel[Ssel==0]=1e-10

        KernelPha=ne.evaluate("f0*(U*l+V*m+W*nn)").astype(self.CType)
        indGauss=np.where(TypeSources==1)[0]

        NGauss=indGauss.size

        T.timeit("2")


        if NGauss>0:
            ang=Gangle[indGauss].reshape((NGauss,1,1,1))
            SigMaj=Gmaj[indGauss].reshape((NGauss,1,1,1))
            SigMin=Gmin[indGauss].reshape((NGauss,1,1,1))
            WaveL=wave
            SminCos=SigMin*np.cos(ang)
            SminSin=SigMin*np.sin(ang)
            SmajCos=SigMaj*np.cos(ang)
            SmajSin=SigMaj*np.sin(ang)
            up=ne.evaluate("U*SminCos-V*SminSin")
            vp=ne.evaluate("U*SmajSin+V*SmajCos")
            const=-(2*(pi**2)*(1/WaveL)**2)#*fudge
            const=const.reshape((1,1,freq.size,1))
            uvp=ne.evaluate("const*((U*SminCos-V*SminSin)**2+(U*SmajSin+V*SmajCos)**2)")
            #KernelPha=ne.evaluate("KernelPha+uvp")
            KernelPha[indGauss,:,:,:]+=uvp[:,:,:,:]

        T.timeit("3")


        LogF=np.log(f)

        T.timeit("3a")
        
        Kernel=ne.evaluate("exp(KernelPha+LogF)")

        T.timeit("4")
        #Kernel=ne.evaluate("f*exp(KernelPha)").astype(self.CType)

        if Kernel.shape[0]>1:
            ColOut=ne.evaluate("sum(Kernel,axis=0)").astype(self.CType)
        else:
            ColOut=Kernel[0]
        T.timeit("5")

        return ColOut
