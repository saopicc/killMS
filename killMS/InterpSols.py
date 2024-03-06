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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import pickle
import numpy as np
import numpy as np
#import pylab
import os
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassInterpol")
from DDFacet.Other.AsyncProcessPool import APP
from DDFacet.Other import Multiprocessing
#from DDFacet.Array import shared_dict
from killMS.Array import NpShared
IdSharedMem=str(int(os.getpid()))+"."
from DDFacet.Other import AsyncProcessPool
from killMS.Other import ClassFitTEC
from killMS.Other import ClassFitAmp
import scipy.ndimage.filters
from pyrap.tables import table
# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# warnings.catch_warnings()
# warnings.filterwarnings('error')
# # ##############################
from killMS.Other.ClassTimeIt import ClassTimeIt

SaveName="last_InterpSols.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SolsFileIn',help='SolfileIn [no default]',default=None)
    group.add_option('--SolsFileOut',help='SolfileOut [no default]',default=None)
    group.add_option('--MSOutFreq',help='The mslist.txt of the ms where frequency needs to be extrapotaler',type=str,default="")
    group.add_option('--NFreqPerMS',help='Number of solution per MS bandwidth',type=int,default=1)
    group.add_option('--NCPU',help='Number of cpu',type=int,default=0)
    
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
    def __init__(self,**kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)

        LMS = [ l.strip() for l in open(self.MSOutFreq).readlines() ]
        self.OutFreqDomains=np.zeros((len(LMS)*self.NFreqPerMS,2),np.float64)
        iFTot=0
        for iMS,MS in enumerate(LMS):
            t=table("%s::SPECTRAL_WINDOW"%MS,ack=False)
            df=t.getcol("CHAN_WIDTH").flat[0]
            fs=t.getcol("CHAN_FREQ").ravel()
            f0,f1=fs[0]-df/2.,fs[-1]+df/2.
            ff=np.linspace(f0,f1,self.NFreqPerMS+1)
            for iF in range(self.NFreqPerMS):
                self.OutFreqDomains[iFTot,0]=ff[iF]
                self.OutFreqDomains[iFTot,1]=ff[iF+1]
                iFTot+=1

        NFreqsOut=self.OutFreqDomains.shape[0]
        self.CentralFreqs=np.mean(self.OutFreqDomains,axis=1)
        
        log.print("Loading %s"%self.SolsFileIn)
        self.DicoFile0=dict(np.load(self.SolsFileIn))
        Dico0=self.DicoFile0
        self.Sols0=self.DicoFile0["Sols"].view(np.recarray)
        
        DicoOut={}
        DicoOut['ModelName']=Dico0['ModelName']
        DicoOut['StationNames']=Dico0['StationNames']
        DicoOut['BeamTimes']=Dico0['BeamTimes']
        DicoOut['SourceCatSub']=Dico0['SourceCatSub']
        DicoOut['ClusterCat']=Dico0['ClusterCat']
        DicoOut['SkyModel']=Dico0['SkyModel']
        DicoOut['FreqDomains']=self.OutFreqDomains
        self.DicoOut=DicoOut
        self.CentralFreqsIn=np.mean(Dico0['FreqDomains'],axis=1)
        
        self.DicoFreqWeights={}
        for iChan in range(self.CentralFreqs.size):
            f=self.CentralFreqs[iChan]
            i0=np.where(self.CentralFreqsIn<=f)[0]
            i1=np.where(self.CentralFreqsIn>f)[0]
            if i0.size>0 and i1.size>0:
                i0=i0[-1]
                i1=i1[0]
                f0=self.CentralFreqsIn[i0]
                f1=self.CentralFreqsIn[i1]
                df=np.abs(f0-f1)
                alpha=1.-(f-f0)/df
                c0=alpha
                c1=1.-alpha
                self.DicoFreqWeights[iChan]={"Type":"Dual",
                                             "Coefs":(c0,c1),
                                             "Index":(i0,i1)}
            else:
                i0=np.argmin(np.abs(self.CentralFreqsIn-f))
                self.DicoFreqWeights[iChan]={"Type":"Single",
                                             "Index":i0}

        
        # for iChan in range(self.CentralFreqs.size):
        #     print 
        #     print self.CentralFreqs[iChan]/1e6
        #     if self.DicoFreqWeights[iChan]["Type"]=="Dual":
        #         i0,i1=self.DicoFreqWeights[iChan]["Index"]
        #         print self.CentralFreqsIn[i0]/1e6,self.CentralFreqsIn[i1]/1e6,self.DicoFreqWeights[iChan]["Coefs"]
        #     else:
        #         i0=self.DicoFreqWeights[iChan]["Index"]
        #         print self.CentralFreqsIn[i0]/1e6
                
                
        nt,_,na,nd,_,_=self.Sols0.G.shape
        SolsOut=np.zeros((nt,),dtype=[("t0",np.float64),("t1",np.float64),("G",np.complex64,(NFreqsOut,na,nd,2,2)),("Stats",np.float32,(NFreqsOut,na,4))])
        
        SolsOut=SolsOut.view(np.recarray)
        SolsOut.t0=self.Sols0.t0
        SolsOut.t1=self.Sols0.t1
        SolsOut.G[...,0,0]=1.
        SolsOut.G[...,1,1]=1.
        self.SolsOut=SolsOut
        self.GOut=NpShared.ToShared("%sGOut"%IdSharedMem,self.SolsOut.G.copy())

        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,affinity=0)

    def Interpol(self):
        APP.startWorkers()
        if "TEC" in self.DicoFile0["SmoothMode"]:
            TECArray=NpShared.ToShared("%sTECArray"%IdSharedMem,self.DicoFile0["SolsTEC"])
            CPhaseArray=NpShared.ToShared("%sCPhaseArray"%IdSharedMem,self.DicoFile0["SolsCPhase"])
            nt,nd,na=TECArray.shape
            iJob=0

            for it in range(nt):
                APP.runJob("InterpolTECTime_%d"%iJob, self.InterpolTECTime, args=(it,))#,serial=True)
                iJob+=1
            workers_res=APP.awaitJobResults("InterpolTECTime*", progress="Interpol TEC")

        iJob=0
        for it in range(nt):
            APP.runJob("InterpolAmpTime_%d"%iJob, self.InterpolAmpTime, args=(it,))#,serial=True)
            iJob+=1
        workers_res=APP.awaitJobResults("InterpolAmpTime*", progress="Interpol Amp")

        # APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()

    def InterpolTECTime(self,iTime):
        GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
        nt,nf,na,nd,_,_=GOut.shape
        TECArray=NpShared.GiveArray("%sTECArray"%IdSharedMem)
        CPhaseArray=NpShared.GiveArray("%sCPhaseArray"%IdSharedMem)
        for iDir in range(nd):
            for iAnt in range(na):
                GThis=TECToZ(TECArray[iTime,iDir,iAnt],CPhaseArray[iTime,iDir,iAnt],self.CentralFreqs.reshape((1,-1)))
                self.GOut[iTime,:,iAnt,iDir,0,0]=GThis
                self.GOut[iTime,:,iAnt,iDir,1,1]=GThis

    def InterpolAmpTime(self,iTime):
        GOut=NpShared.GiveArray("%sGOut"%IdSharedMem)
        nt,nf,na,nd,_,_=GOut.shape
        for iChan in range(nf):
            D=self.DicoFreqWeights[iChan]
            for iDir in range(nd):
                for iAnt in range(na):
                    if D["Type"]=="Dual":
                        i0,i1=D["Index"]
                        c0,c1=D["Coefs"]
                        g=c0*np.abs(self.Sols0.G[iTime,i0,iAnt,iDir,0,0]) + c1*np.abs(self.Sols0.G[iTime,i1,iAnt,iDir,0,0])
                    else:
                        i0=D["Index"]
                        g=np.abs(self.Sols0.G[iTime,i0,iAnt,iDir,0,0])
                        
                    self.GOut[iTime,iChan,iAnt,iDir,0,0]*=g
                    self.GOut[iTime,iChan,iAnt,iDir,1,1]*=g

                
    def Save(self):
        OutFile=self.SolsFileOut
        if not ".npz" in OutFile: OutFile+=".npz"
        
        log.print("  Saving interpolated solution file as: %s"%OutFile)
        self.DicoOut["Sols"]=self.SolsOut
        self.DicoOut["Sols"]["G"][:]=self.GOut[:]
        try:
            np.savez(OutFile,**(self.DicoOut))
        except:
            log.print("There was an exception while using savez")
            log.print("  you should set the environment variable TMPDIR")
            log.print("  to a directory where there is enough space")

            
        NpShared.DelAll("%s"%IdSharedMem)




        
# ############################################        

def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)
    #FileName="killMS.KAFCA.sols.npz"


    if options.SolsFileIn is None or options.SolsFileOut is None:
        raise RuntimeError("You have to specify In/Out solution file names")
    CI=ClassInterpol(**options.__dict__)
    CI.Interpol()
    CI.Save()

def driver():
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)
    
    main(options=options)

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()