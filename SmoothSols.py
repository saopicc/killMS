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
#!/usr/bin/env python
from scipy.optimize import least_squares

import optparse
import pickle
import numpy as np
import numpy as np
import pylab
import os
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassInterpol")
from DDFacet.Other.AsyncProcessPool import APP
from DDFacet.Other import Multiprocessing
#from DDFacet.Array import shared_dict
from killMS.Array import NpShared
IdSharedMem=str(int(os.getpid()))+"."
from DDFacet.Other import AsyncProcessPool
import scipy.ndimage.filters
# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# warnings.catch_warnings()
# warnings.filterwarnings('error')
# # ##############################

SaveName="last_InterPol.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SolsFileIn',help='SolfileIn [no default]',default=None)
    group.add_option('--SolsFileOut',help='SolfileOut [no default]',default=None)
    group.add_option('--InterpMode',help='Interpolation mode TEC and/or Amp [default is %default]',type="str",default="TEC,Amp")
    
    group.add_option('--Amp-SmoothType',help='Interpolation Type for the amplitude [default is %default]',type="str",default="Gauss")
    group.add_option('--Amp-PolyOrder',help='Order of the polynomial to do the amplitude',type="int",default=3)
    group.add_option('--Amp-GaussKernel',help='',type="str",default="1,3")
    group.add_option('--NCPU',help='Number of CPU to use',type="int",default=0)
    opt.add_option_group(group)


    options, arguments = opt.parse_args()
    exec("options.Amp_GaussKernel=%s"%options.Amp_GaussKernel)
    f = open(SaveName,"wb")
    pickle.dump(options,f)

def TECToPhase(TEC,freq):
    K=8.4479745e9
    phase=K*TEC*(1./freq)
    return phase

def TECToZ(TEC,ConstPhase,freq):
    return np.exp(1j*(TECToPhase(TEC,freq)+ConstPhase))



class ClassInterpol():
    def __init__(self,InSolsName,OutSolsName,InterpMode="TEC",PolMode="Scalar",Amp_PolyOrder=3,NCPU=0,
                 Amp_GaussKernel=(0,5), Amp_SmoothType="Poly"):

        self.InSolsName=InSolsName
        self.OutSolsName=OutSolsName
        print>>log,"Loading %s"%self.InSolsName
        self.DicoFile=dict(np.load(self.InSolsName))
        self.Sols=self.DicoFile["Sols"].view(np.recarray)
        self.CentralFreqs=np.mean(self.DicoFile["FreqDomains"],axis=1)

        NTEC=101
        NConstPhase=11
        TECGridAmp=0.1
        TECGrid,CPhase=np.mgrid[-TECGridAmp:TECGridAmp:NTEC*1j,-np.pi:np.pi:NConstPhase*1j]
        Z=TECToZ(TECGrid.reshape((-1,1)),CPhase.reshape((-1,1)),self.CentralFreqs.reshape((1,-1)))
        self.Z=Z
        self.TECGrid,self.CPhase=TECGrid,CPhase

        self.InterpMode=InterpMode
        self.Amp_PolyOrder=Amp_PolyOrder
        self.GOut=NpShared.ToShared("%sGOut"%IdSharedMem,self.Sols.G.copy())
        self.PolMode=PolMode
        self.Amp_GaussKernel=Amp_GaussKernel
        self.Amp_SmoothType=Amp_SmoothType
        self.CalcFreqAmpSystematics()
        self.Sols.G-=self.G0
        if "TEC" in self.InterpMode:
            print>>log, "  Smooth phases using a TEC model"
        if "Amp" in self.InterpMode:
            if Amp_SmoothType=="Poly":
                print>>log, "  Smooth amplitudes using polynomial model of order %i"%self.Amp_PolyOrder
            if Amp_SmoothType=="Gauss":
                print>>log, "  Smooth amplitudes using Gaussian kernel of %s (Time/Freq) bins"%str(Amp_GaussKernel)

        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=NCPU,affinity=0)

    def TECInterPol(self):
        Sols0=self.Sols
        nt,nch,na,nd,_,_=Sols0.G.shape

        for iAnt in range(na):
            for iDir in range(nd):
                for it in range(nt):
                    self.FitThisTEC(it,iAnt,iDir)

    def CalcFreqAmpSystematics(self):
        print>>log, "  Calculating amplitude systematics..."
        Sols0=self.Sols
        nt,nch,na,nd,_,_=Sols0.G.shape
        self.G0=np.zeros((1,nch,na,nd,2,2),np.float32)
        
        for iAnt in range(na):
            for iDir in range(nd):
                G=Sols0.G[:,:,iAnt,iDir,:,:]
                G0=np.mean(np.abs(G),axis=0)
                self.G0[0,:,iAnt,iDir,:,:]=G0.reshape((nch,2,2))
                

    def InterpolParallel(self):
        Sols0=self.Sols
        nt,nch,na,nd,_,_=Sols0.G.shape
        # APP.terminate()
        # APP.shutdown()
        # Multiprocessing.cleanupShm()
        APP.startWorkers()
        iJob=0
        #        for iAnt in [49]:#range(na):
        #            for iDir in [0]:#range(nd):

        print self.InterpMode
        if "CrossTEC" in self.InterpMode:
            for it in range(nt):
                APP.runJob("FitThisCrossTEC_%d"%iJob, self.FitThisCrossTEC, args=(it,))#,serial=True)
                iJob+=1
            workers_res=APP.awaitJobResults("FitThisCrossTEC*", progress="Fit %s"%self.InterpMode)


        for iAnt in range(na):
            for iDir in range(nd):
                APP.runJob("FitThisAmpPhase_%d"%iJob, self.FitThisAmpPhase, args=(iAnt,iDir))#,serial=True)
                iJob+=1
        workers_res=APP.awaitJobResults("FitThisAmpPhase*", progress="Fit %s"%self.InterpMode)


        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()
        # ###########################
        # import pylab
        # op0=np.abs
        # op1=np.angle
        # #for iDir in range(nd):
        # for iAnt in range(40,na):
        #     pylab.clf()
        #     A=op0(self.Sols.G[:,:,iAnt,iDir,0,0])
        #     v0,v1=0,A.max()
        #     pylab.subplot(2,3,1)
        #     pylab.imshow(op0(self.Sols.G[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto",vmin=v0,vmax=v1)
        #     pylab.title("Raw Solution (Amp)")
        #     pylab.xlabel("Freq bin")
        #     pylab.ylabel("Time bin")

        #     pylab.subplot(2,3,2)
        #     pylab.imshow(op0(self.GOut[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto",vmin=v0,vmax=v1)
        #     pylab.title("Smoothed Solution (Amp)")
        #     pylab.xlabel("Freq bin")
        #     pylab.ylabel("Time bin")

        #     pylab.subplot(2,3,3)
        #     pylab.imshow(op0(self.Sols.G[:,:,iAnt,iDir,0,0])-op0(self.GOut[:,:,iAnt,iDir,0,0]),interpolation="nearest",
        #                  aspect="auto",vmin=v0,vmax=v1)
        #     pylab.xlabel("Freq bin")
        #     pylab.ylabel("Time bin")
        #     pylab.title("Residual (Amp)")
        #     #pylab.colorbar()
        #     A=op1(self.Sols.G[:,:,iAnt,iDir,0,0])
        #     v0,v1=A.min(),A.max()
        #     pylab.subplot(2,3,4)
        #     pylab.imshow(op1(self.Sols.G[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto",vmin=v0,vmax=v1)
        #     pylab.title("Raw Solution (Phase)")
        #     pylab.xlabel("Freq bin")
        #     pylab.ylabel("Time bin")

        #     pylab.subplot(2,3,5)
        #     pylab.imshow(op1(self.GOut[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto",vmin=v0,vmax=v1)
        #     pylab.title("Smoothed Solution (Phase)")
        #     pylab.xlabel("Freq bin")
        #     pylab.ylabel("Time bin")

        #     pylab.subplot(2,3,6)
        #     pylab.imshow(op1(self.Sols.G[:,:,iAnt,iDir,0,0])-op1(self.GOut[:,:,iAnt,iDir,0,0]),
        #                  interpolation="nearest",aspect="auto",vmin=v0,vmax=v1)
        #     pylab.title("Residual (Phase)")
        #     pylab.xlabel("Freq bin")
        #     pylab.ylabel("Time bin")

        #     pylab.suptitle("(iAnt, iDir) = (%i, %i)"%(iAnt,iDir))
        #     pylab.tight_layout()
        #     pylab.draw()
        #     pylab.show()#False)
        #     pylab.pause(0.1)
        #     #stop

    

    def FitThisCrossTEC(self,it):
        nt,nch,na,nd,_,_=self.Sols.G.shape
        for iDir in range(nd):
            gz,TEC,CPhase=self.FitThisCrossTECTime(it,iDir)
            GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
            GOut[it,:,:,iDir,0,0]=gz
            GOut[it,:,:,iDir,1,1]=gz

        
    def FitThisCrossTECTime(self,it,iDir):
        GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
        nt,nch,na,nd,_,_=self.Sols.G.shape

        TEC0CPhase0=np.zeros((2,na),np.float32)
        for iAnt in range(na):
            _,t0,c0=self.FitThisTECTime(it,iAnt,iDir)
            TEC0CPhase0[0,iAnt]=t0
            TEC0CPhase0[1,iAnt::]=c0
        
        G=GOut[it,:,:,iDir,0,0].T.copy()
        A0,A1=np.mgrid[0:na,0:na]
        gg_meas=G[A0.ravel(),:]*G[A1.ravel(),:].conj()
        gg_meas_reim=np.array([gg_meas.real,gg_meas.imag]).ravel()[::11]
        # for ibl in range(gg_meas.shape[0])[::-1]:
        #     import pylab
        #     pylab.clf()
        #     pylab.subplot(2,1,1)
        #     pylab.scatter(self.CentralFreqs,np.abs(gg_meas[ibl]))
        #     pylab.ylim(0,5)
        #     pylab.subplot(2,1,2)
        #     pylab.scatter(self.CentralFreqs,np.angle(gg_meas[ibl]))
        #     pylab.ylim(-np.pi,np.pi)
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        
        def _f_resid(TecConst,A0,A1,ggmeas):
            TEC,CPhase=TecConst.reshape((2,na))
            GThis=TECToZ(TEC.reshape((-1,1)),CPhase.reshape((-1,1)),self.CentralFreqs.reshape((1,-1)))
            gg_pred=GThis[A0.ravel(),:]*GThis[A1.ravel(),:].conj()
            gg_pred_reim=np.array([gg_pred.real,gg_pred.imag]).ravel()[::11]
            r=(ggmeas-gg_pred_reim).ravel()
            #return np.angle((ggmeas-gg_pred).ravel())
            #print np.mean(np.abs(r))
            return r
        #print _f_resid(TEC0CPhase0,A0,A1,ggmeas)

        print "start"
        Sol=least_squares(_f_resid, TEC0CPhase0.ravel(), args=(A0,A1,gg_meas_reim),ftol=1e-2,gtol=1e-2,xtol=1e-2)
        print "ok",it,iDir
        TEC,CPhase=Sol.x.reshape((2,na))

        TEC-=TEC[0]
        CPhase-=CPhase[0]
        GThis=np.abs(GOut[it,:,:,iDir,0,0]).T*TECToZ(TEC.reshape((-1,1)),CPhase.reshape((-1,1)),self.CentralFreqs.reshape((1,-1)))

        return GThis.T,TEC,CPhase
        # # ###########################
        # TEC0,CPhase0=TEC0CPhase0
        # GThis0=TECToZ(TEC0.reshape((-1,1)),CPhase0.reshape((-1,1)),self.CentralFreqs.reshape((1,-1)))

        # for iAnt in range(na):
        #     print "!!!!!!!!!!!!!!",iAnt,iDir
        #     ga=GOut[it,:,iAnt,iDir,0,0]
        #     ge=GThis[iAnt,:]
        #     ge0=GThis0[iAnt,:]
        #     #if iAnt==0: continue
        #     #f=np.linspace(self.CentralFreqs.min(),self.CentralFreqs.max(),100)
        #     #ztec=TECToZ(TECGrid.ravel()[iTec],CPhase.ravel()[iTec],f)
        #     import pylab
        #     pylab.clf()
        #     pylab.subplot(1,2,1)
        #     pylab.scatter(self.CentralFreqs,np.abs(ga),color="black")
        #     pylab.plot(self.CentralFreqs,np.abs(ge),ls=":",color="black")
        #     pylab.subplot(1,2,2)
        #     pylab.scatter(self.CentralFreqs,np.angle(ga),color="black")
        #     pylab.plot(self.CentralFreqs,np.angle(ge),ls=":",color="black")
        #     pylab.plot(self.CentralFreqs,np.angle(ge0),ls=":",color="red")
        #     #pylab.plot(f,np.angle(ztec),ls=":",color="black")
        #     pylab.ylim(-np.pi,np.pi)
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        # # ###############################
        

        
    def FitThisAmpPhase(self,iAnt,iDir):
        nt,nch,na,nd,_,_=self.Sols.G.shape
        if "TEC" in self.InterpMode:
            for it in range(nt):
                gz,t0,c0=self.FitThisTECTime(it,iAnt,iDir)
                GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
                GOut[it,:,iAnt,iDir,0,0]=gz
                GOut[it,:,iAnt,iDir,1,1]=gz

        if "Amp" in self.InterpMode:
            if self.Amp_SmoothType=="Poly":
                for it in range(nt):
                    self.FitThisAmpTimePoly(it,iAnt,iDir)
            elif self.Amp_SmoothType=="Gauss":
                self.GaussSmoothAmp(iAnt,iDir)

    def FitThisTECTime(self,it,iAnt,iDir):
        GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
        g=GOut[it,:,iAnt,iDir,0,0]
        g0=g/np.abs(g)
        W=np.ones(g0.shape,np.float32)
        W[g==1.]=0
        Z=self.Z
        for iTry in range(5):
            R=(g0.reshape((1,-1))-Z)*W.reshape((1,-1))
            Chi2=np.sum(np.abs(R)**2,axis=1)
            iTec=np.argmin(Chi2)
            rBest=R[iTec]
            if np.max(np.abs(rBest))==0: break
            Sig=np.sum(np.abs(rBest*W))/np.sum(W)
            ind=np.where(np.abs(rBest*W)>5.*Sig)[0]
            if ind.size==0: break
            W[ind]=0

            # gz=TECToZ(TECGrid.ravel()[iTec],CPhase.ravel()[iTec],self.CentralFreqs)
            # import pylab
            # pylab.clf()
            # pylab.subplot(2,1,1)
            # pylab.scatter(self.CentralFreqs,rBest)
            # pylab.scatter(self.CentralFreqs[ind],rBest[ind],color="red")
            # pylab.subplot(2,1,2)
            # pylab.scatter(self.CentralFreqs,rBest)
            # pylab.scatter(self.CentralFreqs[ind],rBest[ind],color="red")
            # pylab.draw()
            # pylab.show()


        
        # # ###########################
        # print iAnt,iDir
        # if iAnt==0: return
        # f=np.linspace(self.CentralFreqs.min(),self.CentralFreqs.max(),100)
        # ztec=TECToZ(TECGrid.ravel()[iTec],CPhase.ravel()[iTec],f)
        # import pylab
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.scatter(self.CentralFreqs,np.abs(g),color="black")
        # pylab.plot(self.CentralFreqs,np.abs(gz),ls=":",color="black")
        # pylab.plot(self.CentralFreqs,np.abs(gz)-np.abs(g),ls=":",color="red")
        # pylab.subplot(1,2,2)
        # pylab.scatter(self.CentralFreqs,np.angle(g),color="black")
        # pylab.plot(self.CentralFreqs,np.angle(gz),ls=":",color="black")
        # pylab.plot(self.CentralFreqs,np.angle(gz)-np.angle(g),ls=":",color="red")
        # #pylab.plot(f,np.angle(ztec),ls=":",color="black")
        # pylab.ylim(-np.pi,np.pi)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # # ###############################

        t0=self.TECGrid.ravel()[iTec]
        c0=self.CPhase.ravel()[iTec]
    
        gz=np.abs(g)*TECToZ(t0,c0,self.CentralFreqs)
        return gz,t0,c0


        
    def FitThisAmpTimePoly(self,it,iAnt,iDir):
        GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
        g=GOut[it,:,iAnt,iDir,0,0]
        g0=np.abs(g)
        
        W=np.ones(g0.shape,np.float32)
        W[g0==1.]=0
        if np.count_nonzero(W)<self.Amp_PolyOrder*3: return

        for iTry in range(5):
            if np.max(W)==0: return
            z = np.polyfit(self.CentralFreqs, g0, self.Amp_PolyOrder,w=W)
            p = np.poly1d(z)
            gz=p(self.CentralFreqs)*g/np.abs(g)
            rBest=(g0-gz)
            if np.max(np.abs(rBest))==0: break
            Sig=np.sum(np.abs(rBest*W))/np.sum(W)
            ind=np.where(np.abs(rBest*W)>5.*Sig)[0]
            if ind.size==0: break
            W[ind]=0

        GOut[it,:,iAnt,iDir,0,0]=gz
        GOut[it,:,iAnt,iDir,1,1]=gz

    def GaussSmoothAmp(self,iAnt,iDir):
        GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
        g=GOut[:,:,iAnt,iDir,0,0]
        g0=np.abs(g)

        
        sg0=scipy.ndimage.filters.gaussian_filter(g0,self.Amp_GaussKernel)

        gz=sg0*g/np.abs(g)

        GOut[:,:,iAnt,iDir,0,0]=gz
        GOut[:,:,iAnt,iDir,1,1]=gz

    def Save(self):
        OutFile=self.OutSolsName
        if not ".npz" in OutFile: OutFile+=".npz"
        print>>log,"  Saving interpolated solution file as: %s"%OutFile
        self.DicoFile["Sols"]["G"][...]=self.GOut[:]
        np.savez(OutFile,**(self.DicoFile))
        NpShared.DelAll("%sGOut"%IdSharedMem)

# ############################################        

def test():
    FileName="L401839.killms_f_ap_deep.merged.npz"
    OutFile="TestMerge.Interp.npz"
    CI=ClassInterpol(FileName,OutFile)
    CI.InterpolParallel()
    return CI.Save()

def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)
    #FileName="killMS.KAFCA.sols.npz"


    if options.SolsFileIn is None or options.SolsFileOut is None:
        raise RuntimeError("You have to specify In/Out solution file names")
    CI=ClassInterpol(options.SolsFileIn,
                     options.SolsFileOut,
                     InterpMode=options.InterpMode,
                     Amp_PolyOrder=options.Amp_PolyOrder,
                     Amp_GaussKernel=options.Amp_GaussKernel, Amp_SmoothType=options.Amp_SmoothType,
                     NCPU=options.NCPU)
    CI.InterpolParallel()
    CI.Sols.G+=CI.G0

    CI.Save()


if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)
    
    main(options=options)
