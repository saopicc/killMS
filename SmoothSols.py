#!/usr/bin/env python

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
from killMS2.Array import NpShared
IdSharedMem=str(int(os.getpid()))+"."
from DDFacet.Other import AsyncProcessPool

SaveName="last_InterPol.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SolsFileIn',help='Solfile [no default]',default=None)
    group.add_option('--SolsFileOut',help='Solfile [no default]',default=None)
    group.add_option('--InterpMode',help='Interpolation mode TEC and/or Amp [default is %default]',type="str",default="TEC,Amp")
    group.add_option('--PolyOrder',help='Order of the polynomial to do the amplitude',type="int",default=3)
    group.add_option('--NCPU',help='Number of CPU to use',type="int",default=0)
    opt.add_option_group(group)


    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)

def TECToPhase(TEC,freq):
    K=8.4479745e9
    phase=K*TEC*(1./freq)
    return phase

def TECToZ(TEC,ConstPhase,freq):
    return np.exp(1j*(TECToPhase(TEC,freq)+ConstPhase))

class ClassInterpol():
    def __init__(self,InSolsName,OutSolsName,InterpMode="TEC",PolMode="Scalar",PolyOrder=3,NCPU=0):

        self.InSolsName=InSolsName
        self.OutSolsName=OutSolsName
        print>>log,"Loading %s"%self.InSolsName
        self.DicoFile=dict(np.load(self.InSolsName))
        self.Sols=self.DicoFile["Sols"].view(np.recarray)
        self.CentralFreqs=np.mean(self.DicoFile["FreqDomains"],axis=1)
        self.InterpMode=InterpMode
        self.PolyOrder=PolyOrder
        self.GOut=NpShared.ToShared("%sGOut"%IdSharedMem,self.Sols.G.copy())
        self.PolMode=PolMode

        if "TEC" in self.InterpMode:
            print>>log, "  Smooth phases using a TEC model"
        if "Amp" in self.InterpMode:
            print>>log, "  Smooth amplitudes using polynomial model of order %i"%self.PolyOrder

        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=NCPU,affinity=0)

    def TECInterPol(self):
        Sols0=self.Sols
        nt,nch,na,nd,_,_=Sols0.G.shape

        for iAnt in range(na):
            for iDir in range(nd):
                for it in range(nt):
                    self.FitThisTEC(it,iAnt,iDir)


    def InterpolParallel(self):
        Sols0=self.Sols
        nt,nch,na,nd,_,_=Sols0.G.shape
        # APP.terminate()
        # APP.shutdown()
        # Multiprocessing.cleanupShm()
        APP.startWorkers()
        iJob=0
        for iAnt in range(na):
            for iDir in range(nd):
                APP.runJob("FitThisTEC_%d"%iJob, self.FitThisTEC, args=(iAnt,iDir))#,serial=True)
                iJob+=1
        workers_res=APP.awaitJobResults("FitThisTEC*", progress="Fit %s"%self.InterpMode)
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()
        NpShared.DelAll("%sGOut"%IdSharedMem)
        # ###########################
        # import pylab
        # op0=np.abs
        # op1=np.angle
        # for iDir in range(nd):
        #     pylab.clf()
        #     pylab.subplot(2,3,1)
        #     pylab.imshow(op0(self.Sols.G[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto")
        #     pylab.subplot(2,3,2)
        #     pylab.imshow(op0(self.GOut[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto")
        #     pylab.subplot(2,3,3)
        #     pylab.imshow(op0(self.Sols.G[:,:,iAnt,iDir,0,0])-op0(self.GOut[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto")
        #     pylab.subplot(2,3,4)
        #     pylab.imshow(op1(self.Sols.G[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto")
        #     pylab.subplot(2,3,5)
        #     pylab.imshow(op1(self.GOut[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto")
        #     pylab.subplot(2,3,6)
        #     pylab.imshow(op1(self.Sols.G[:,:,iAnt,iDir,0,0])-op1(self.GOut[:,:,iAnt,iDir,0,0]),interpolation="nearest",aspect="auto")
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     stop


    def FitThisTEC(self,iAnt,iDir):
        nt,nch,na,nd,_,_=self.Sols.G.shape
        for it in range(nt):
            if "TEC" in self.InterpMode:
                self.FitThisTECTime(it,iAnt,iDir)
            if "Amp" in self.InterpMode:
                self.FitThisAmpTime(it,iAnt,iDir)

    def FitThisTECTime(self,it,iAnt,iDir):
        GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
        g=GOut[it,:,iAnt,iDir,0,0]
        g0=g/np.abs(g)
        NTEC=101
        NConstPhase=21
        TECGridAmp=0.1
        TECGrid,CPhase=np.mgrid[-TECGridAmp:TECGridAmp:NTEC*1j,-np.pi:np.pi:NConstPhase*1j]
        Z=TECToZ(TECGrid.reshape((-1,1)),CPhase.reshape((-1,1)),self.CentralFreqs.reshape((1,-1)))
        W=np.ones(g0.shape,np.float32)
        W[g==1.]=0
        R=(g0.reshape((1,-1))-Z)*W.reshape((1,-1))
        Chi2=np.sum(np.abs(R)**2,axis=1)
        iTec=np.argmin(Chi2)
        
        
        gz=np.abs(g)*TECToZ(TECGrid.ravel()[iTec],CPhase.ravel()[iTec],self.CentralFreqs)
        
        GOut[it,:,iAnt,iDir,0,0]=gz
        GOut[it,:,iAnt,iDir,1,1]=gz

        # ###########################
        # f=np.linspace(self.CentralFreqs.min(),self.CentralFreqs.max(),100)
        # ztec=TECToZ(TECGrid.ravel()[iTec],CPhase.ravel()[iTec],f)
        # import pylab
        # pylab.clf()
        # pylab.scatter(self.CentralFreqs,np.angle(g),color="black")
        # pylab.plot(f,np.angle(ztec),ls=":",color="black")
        # pylab.ylim(-np.pi,np.pi)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

    def FitThisAmpTime(self,it,iAnt,iDir):
        GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
        g=GOut[it,:,iAnt,iDir,0,0]
        g0=np.abs(g)


        z = np.polyfit(self.CentralFreqs, g0, self.PolyOrder)
        p = np.poly1d(z)
        
        gz=p(self.CentralFreqs)*g/np.abs(g)

        GOut[it,:,iAnt,iDir,0,0]=gz
        GOut[it,:,iAnt,iDir,1,1]=gz

    def Save(self):
        OutFile=self.OutSolsName
        if not ".npz" in OutFile: OutFile+=".npz"
        print>>log,"  Saving interpolated solution file as: %s"%OutFile
        self.DicoFile["Sols"]["G"][...]=self.GOut[:]
        np.savez(OutFile,**(self.DicoFile))

# ############################################        

def test():
    FileName="TestMerge.npz"
    CI=ClassInterpol(FileName,"TestMergeOut")
    CI.InterpolParallel()
    OutFile="TestMerge.Interpol.npz"
    CI.Save(OutFile)

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
                     PolyOrder=options.PolyOrder,
                     NCPU=options.NCPU)
    CI.InterpolParallel()
    CI.Save()


if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)


    main(options=options)
