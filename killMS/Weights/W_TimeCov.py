from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
from pyrap.tables import table
import sys
from DDFacet.Other import logger
log=logger.getLogger("DynSpecMS")
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
import numpy as np
from astropy.time import Time
from astropy import constants as const
import os
from killMS.Other import reformat
from DDFacet.Other import AsyncProcessPool
import glob
from astropy.io import fits
from astropy.wcs import WCS
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
import scipy.stats
from killMS.Array import ModLinAlg

# # ##############################
# # Catch numpy warning
# import numpy as np
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################


def AngDist(ra0,ra1,dec0,dec1):
    AC=np.arccos
    C=np.cos
    S=np.sin
    D=S(dec0)*S(dec1)+C(dec0)*C(dec1)*C(ra0-ra1)
    
    if type(D).__name__=="ndarray":
        D[D>1.]=1.
        D[D<-1.]=-1.
    else:
        if D>1.: D=1.
        if D<-1.: D=-1.
    return AC(D)

def test():
    CW=ClassCovMat(ListMSName=["1563189318_sdp_l0-A3528S_corr.ms.tsel2"],
                   FileCoords="a3528-beam_ds9.Cyril.WeightKMS_fixBeam.npy.ClusterCat.npy",
                   #ListMSName=["0000.MS"],
                   #FileCoords="Target.txt.ClusterCat.npy",
                   ColName="CORRECTED_DATA",
                   ModelName="DDF_PREDICT",
                   UVRange=[0.1,1000.], 
                   ColWeights=None, 
                   SolsName=None,
                   SolsDir=None,
                   NCPU=0,
                   BeamModel=None,
                   BeamNBand=1)   
    CW.StackAll()

class ClassCovMat(object):
    def __init__(self,
                 ListMSName=None,
                 ColName="DATA",
                 ModelName="PREDICT_KMS",
                 UVRange=[1.,1000.], 
                 ColWeights=None, 
                 SolsName=None,
                 FileCoords="Transient_LOTTS.csv",
                 SolsDir=None,
                 NCPU=1,
                 BeamModel=None,
                 BeamNBand=1):

        self.ColName    = ColName
        self.ModelName  = ModelName
        self.ColWeights = ColWeights
        self.BeamNBand  = BeamNBand
        self.UVRange    = UVRange
        self.Mode="Spec"
        self.SolsName=SolsName
        self.NCPU=NCPU
        self.BeamModel=BeamModel
        self.StepFreq=2
        self.StepTime=10
        
        if ListMSName is None:
            print(ModColor.Str("WORKING IN REPLOT MODE"), file=log)
            self.Mode="Plot"
            
            
        self.SolsDir=SolsDir

        self.FileCoords=FileCoords
        
        self.ListMSName = sorted(ListMSName)#[0:2]
        self.nMS         = len(self.ListMSName)
        self.OutName    = self.ListMSName[0].split("/")[-1].split("_")[0]
        self.ReadMSInfos()
        self.InitFromCatalog()


            
    def InitFromCatalog(self):

        FileCoords=self.FileCoords
        dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),('Type','S200')]
        #FileCoords="Transient_LOTTS.csv"
        
        self.PosArray=np.load(FileCoords)
        self.PosArray=self.PosArray.view(np.recarray)

        self.NDirSelected=self.PosArray.shape[0]

        self.NDir=self.PosArray.shape[0]



        
        self.DicoDATA = shared_dict.create("DATA")
        self.DicoGrids = shared_dict.create("Grids")
        
        dChan=np.min([self.StepFreq,self.NChan])
        self.DicoGrids["DomainEdges_Freq"] = np.int64(np.linspace(0,self.NChan,int(self.NChan/dChan)+1))
        #self.DicoGrids["DomainEdges_Time"] = np.int64(np.linspace(0,self.NTimes-1,int(self.NTimes/self.StepTime)+1))
        DT=self.times.max()-self.times.min()
        DTSol=np.min([self.StepTime*self.dt,DT])
        self.DicoGrids["DomainEdges_Time"] = np.linspace(self.times.min()-1e-6,self.times.max()+1e-6,int(DT/DTSol)+1)

        self.DicoGrids["GridC2"] = np.zeros((self.DicoGrids["DomainEdges_Time"].size-1,self.DicoGrids["DomainEdges_Freq"].size-1, self.na,self.na), np.complex128)
        self.DicoGrids["GridC"] = np.zeros((self.DicoGrids["DomainEdges_Time"].size-1,self.DicoGrids["DomainEdges_Freq"].size-1, self.na,self.na), np.complex128)
        
        log.print("  DomainEdges_Freq: %s"%(str(self.DicoGrids["DomainEdges_Freq"])))
        log.print("  DomainEdges_Time: %s"%(str(self.DicoGrids["DomainEdges_Time"])))
        

        self.DoJonesCorr_kMS =False
        self.DicoJones=None
        if self.SolsName:
            self.DoJonesCorr_kMS=True
            self.DicoJones_kMS=shared_dict.create("DicoJones_kMS")

        self.DoJonesCorr_Beam=False
        if self.BeamModel:
            self.DoJonesCorr_Beam=True
            self.DicoJones_Beam=shared_dict.create("DicoJones_Beam")


        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,affinity=0)
        APP.startWorkers()



    def ReadMSInfos(self):
        DicoMSInfos = {}

        MSName=self.ListMSName[0]
        t0  = table(MSName, ack=False)
        tf0 = table("%s::SPECTRAL_WINDOW"%self.ListMSName[0], ack=False)
        self.ChanWidth = tf0.getcol("CHAN_WIDTH").ravel()[0]
        tf0.close()
        self.times = np.sort(np.unique(t0.getcol("TIME")))
        self.dt=(self.times[1::]-self.times[0:-1]).min()
        t0.close()

        ta = table("%s::ANTENNA"%self.ListMSName[0], ack=False)
        self.na=ta.getcol("POSITION").shape[0]
        ta.close()
        
        tField = table("%s::FIELD"%MSName, ack=False)
        self.ra0, self.dec0 = tField.getcol("PHASE_DIR").ravel() # radians!
        if self.ra0<0.: self.ra0+=2.*np.pi
        tField.close()

        pBAR = ProgressBar(Title="Reading metadata")
        pBAR.render(0, self.nMS)
   
        #for iMS, MSName in enumerate(sorted(self.ListMSName)):
        for iMS, MSName in enumerate(self.ListMSName):
            try:
                t = table(MSName, ack=False)
            except Exception as e:
                s = str(e)
                DicoMSInfos[iMS] = {"Readable": False,
                                    "Exception": s}
                pBAR.render(iMS+1, self.nMS)
                continue

            if self.ColName not in t.colnames():
                DicoMSInfos[iMS] = {"Readable": False,
                                    "Exception": "Missing Data colname %s"%self.ColName}
                pBAR.render(iMS+1, self.nMS)
                continue

            if self.ColWeights and (self.ColWeights not in t.colnames()):
                DicoMSInfos[iMS] = {"Readable": False,
                                    "Exception": "Missing Weights colname %s"%self.ColWeights}
                pBAR.render(iMS+1, self.nMS)
                continue

            
            if  self.ModelName and (self.ModelName not in t.colnames()):
                DicoMSInfos[iMS] = {"Readable": False,
                                    "Exception": "Missing Model colname %s"%self.ModelName}
                pBAR.render(iMS+1, self.nMS)
                continue
            

            tf = table("%s::SPECTRAL_WINDOW"%MSName, ack=False)
            ThisTimes = np.unique(t.getcol("TIME"))
            if not np.allclose(ThisTimes, self.times):
                raise ValueError("should have the same times")
            DicoMSInfos[iMS] = {"MSName": MSName,
                                "ChanFreq":   tf.getcol("CHAN_FREQ").ravel(),  # Hz
                                "ChanWidth":  tf.getcol("CHAN_WIDTH").ravel(), # Hz
                                "times":      ThisTimes,
                                "startTime":  Time(ThisTimes[0]/(24*3600.), format='mjd', scale='utc').isot,
                                "stopTime":   Time(ThisTimes[-1]/(24*3600.), format='mjd', scale='utc').isot,
                                "deltaTime":  (ThisTimes[-1] - ThisTimes[0])/3600., # h
                                "Readable":   True}
            if DicoMSInfos[iMS]["ChanWidth"][0] != self.ChanWidth:
                raise ValueError("should have the same chan width")
            pBAR.render(iMS+1, self.nMS)
            
        for iMS in range(self.nMS):
            if not DicoMSInfos[iMS]["Readable"]:
                print(ModColor.Str("Problem reading %s"%MSName), file=log)
                print(ModColor.Str("   %s"%DicoMSInfos[iMS]["Exception"]), file=log)
                

        t.close()
        tf.close()
        self.DicoMSInfos = DicoMSInfos
        self.FreqsAll    = np.array([DicoMSInfos[iMS]["ChanFreq"] for iMS in list(DicoMSInfos.keys()) if DicoMSInfos[iMS]["Readable"]])
        self.Freq_minmax = np.min(self.FreqsAll), np.max(self.FreqsAll)
        self.NTimes      = self.times.size
        f0, f1           = self.Freq_minmax
        self.NChan       = int((f1 - f0)/self.ChanWidth) + 1

        # Fill properties
        self.tStart = DicoMSInfos[0]["startTime"]
        self.tStop  = DicoMSInfos[0]["stopTime"] 
        self.fMin   = self.Freq_minmax[0]
        self.fMax   = self.Freq_minmax[1]

        self.iCurrentMS=0


    
    def LoadNextMS(self):
        iMS=self.iCurrentMS
        if not self.DicoMSInfos[iMS]["Readable"]: 
            print("Skipping [%i/%i]: %s"%(iMS+1, self.nMS, self.ListMSName[iMS]), file=log)
            self.iCurrentMS+=1
            return "NotRead"
        print("Reading [%i/%i]: %s"%(iMS+1, self.nMS, self.ListMSName[iMS]), file=log)

        MSName=self.ListMSName[self.iCurrentMS]
        
        t         = table(MSName, ack=False)
        data      = t.getcol(self.ColName)
        flag   = t.getcol("FLAG")
        flag[data==0]=1
        
        if self.ModelName:
            print("  Substracting %s from %s"%(self.ModelName,self.ColName), file=log)
            dp=t.getcol(self.ModelName)
            flag[dp==0]=1
            data-=dp
            data_p=dp
            #del(dp)

        data[flag==1]=0
        data_p[flag==1]=0
        
        if self.ColWeights:
            print("  Reading weight column %s"%(self.ColWeights), file=log)
            weights   = t.getcol(self.ColWeights)
        else:
            nrow,nch,_=data.shape
            weights=np.ones((nrow,nch),np.float32)
        
        times  = t.getcol("TIME")
        A0, A1 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        u, v, w = t.getcol("UVW").T
        t.close()
        d = np.sqrt(u**2 + v**2 + w**2)
        uv0, uv1         = np.array(self.UVRange) * 1000
        indUV = np.where( (d<uv0)|(d>uv1) )[0]
        flag[indUV, :, :] = 1 # flag according to UV selection
        data[flag] = 0 # put down to zeros flagged visibilities

        f0, f1           = self.Freq_minmax
        nch  = self.DicoMSInfos[iMS]["ChanFreq"].size

        # Considering another position than the phase center
        u0 = u.reshape( (-1, 1, 1) )
        v0 = v.reshape( (-1, 1, 1) )
        w0 = w.reshape( (-1, 1, 1) )
        self.DicoDATA["iMS"]=self.iCurrentMS
        self.DicoDATA["MSName"]=self.ListMSName[self.iCurrentMS]
        self.DicoDATA["data"]=data
        self.DicoDATA["data_p"]=data_p

        self.DicoDATA["WOUT"]=np.zeros((data.shape[0],data.shape[1]),np.float64)
        self.DicoDATA["weights"]=weights
        self.DicoDATA["flag"]=flag
        self.DicoDATA["times"]=times
        self.DicoDATA["A0"]=A0
        self.DicoDATA["A1"]=A1
        self.DicoDATA["u"]=u0
        self.DicoDATA["v"]=v0
        self.DicoDATA["w"]=w0
        self.DicoDATA["uniq_times"]=np.unique(self.DicoDATA["times"])

        dstat0=data[:,:,0][flag[:,:,0]==0]
        dstat1=data[:,:,-1][flag[:,:,-1]==0]
        self.DicoDATA["RMS"]=scipy.stats.median_abs_deviation(np.concatenate([dstat0,dstat1],axis=0).ravel(),scale="normal")
        log.print("    estimated RMS=%f"%self.DicoDATA["RMS"])
            
        if self.DoJonesCorr_kMS or self.DoJonesCorr_Beam:
            self.setJones()
        self.iCurrentMS+=1

    def setJones(self):
        from DDFacet.Data import ClassJones
        from DDFacet.Data import ClassMS

        SolsName=self.SolsName
        if "[" in SolsName:
            SolsName=SolsName.replace("[","")
            SolsName=SolsName.replace("]","")
            SolsName=SolsName.split(",")
        GD={"Beam":{"Model":self.BeamModel,
                    "LOFARBeamMode":"A",
                    "DtBeamMin":5.,
                    "NBand":self.BeamNBand,
                    "CenterNorm":1},
            "Image":{"PhaseCenterRADEC":None},
            "DDESolutions":{"DDSols":SolsName,
                            "SolsDir":self.SolsDir,
                            "GlobalNorm":None,
                            "JonesNormList":"AP"},
            "Cache":{"Dir":""}
            }
        print("Reading Jones matrices solution file:", file=log)

        ms=ClassMS.ClassMS(self.DicoMSInfos[self.iCurrentMS]["MSName"],GD=GD,DoReadData=False,)
        JonesMachine = ClassJones.ClassJones(GD, ms, CacheMode=False)
        JonesMachine.InitDDESols(self.DicoDATA)
        #JJ=JonesMachine.MergeJones(self.DicoDATA["killMS"]["Jones"],self.DicoDATA["Beam"]["Jones"])
        # import killMS.Data.ClassJonesDomains
        # DomainMachine=killMS.Data.ClassJonesDomains.ClassJonesDomains()
        # if "killMS" in self.DicoDATA.keys():
        #     self.DicoDATA["killMS"]["Jones"]["FreqDomain"]=self.DicoDATA["killMS"]["Jones"]["FreqDomains"]
        # if "Beam" in self.DicoDATA.keys():
        #     self.DicoDATA["Beam"]["Jones"]["FreqDomain"]=self.DicoDATA["Beam"]["Jones"]["FreqDomains"]
        # if "killMS" in self.DicoDATA.keys() and "Beam" in self.DicoDATA.keys():
        #     JonesSols=DomainMachine.MergeJones(self.DicoDATA["killMS"]["Jones"],self.DicoDATA["Beam"]["Jones"])
        # elif "killMS" in self.DicoDATA.keys() and not ("Beam" in self.DicoDATA.keys()):
        #     JonesSols=self.DicoDATA["killMS"]["Jones"]
        # elif not("killMS" in self.DicoDATA.keys()) and "Beam" in self.DicoDATA.keys():
        #     JonesSols=self.DicoDATA["Beam"]["Jones"]

        #self.DicoJones["G"]=np.swapaxes(self.NormJones(JonesSols["Jones"]),1,3) # Normalize Jones matrices

        if self.DoJonesCorr_kMS:
            JonesSols=self.DicoDATA["killMS"]["Jones"]
            self.DicoJones_kMS["G"]=np.swapaxes(JonesSols["Jones"],1,3) # Normalize Jones matrices
            self.DicoJones_kMS['tm']=(JonesSols["t0"]+JonesSols["t1"])/2.
            self.DicoJones_kMS['ra']=JonesMachine.ClusterCat['ra']
            self.DicoJones_kMS['dec']=JonesMachine.ClusterCat['dec']
            self.DicoJones_kMS['FreqDomains']=JonesSols['FreqDomains']
            self.DicoJones_kMS['FreqDomains_mean']=np.mean(JonesSols['FreqDomains'],axis=1)
            self.DicoJones_kMS['IDJones']=np.zeros((self.NDir,),np.int32)
            for iDir in range(self.NDir):
                ra=self.PosArray.ra[iDir]
                dec=self.PosArray.dec[iDir]
                self.DicoJones_kMS['IDJones'][iDir]=np.argmin(AngDist(ra,self.DicoJones_kMS['ra'],dec,self.DicoJones_kMS['dec']))

        if self.DoJonesCorr_Beam:
            JonesSols = JonesMachine.GiveBeam(np.unique(self.DicoDATA["times"]), quiet=True,RaDec=(self.PosArray.ra,self.PosArray.dec))
            self.DicoJones_Beam["G"]=np.swapaxes(JonesSols["Jones"],1,3) # Normalize Jones matrices
            self.DicoJones_Beam['tm']=(JonesSols["t0"]+JonesSols["t1"])/2.
            self.DicoJones_Beam['ra']=self.PosArray.ra
            self.DicoJones_Beam['dec']=self.PosArray.dec
            self.DicoJones_Beam['FreqDomains']=JonesSols['FreqDomains']
            self.DicoJones_Beam['FreqDomains_mean']=np.mean(JonesSols['FreqDomains'],axis=1)

        
        # from DDFacet.Data import ClassLOFARBeam
        # GD,D={},{}
        # D["LOFARBeamMode"]="A"
        # D["DtBeamMin"]=5
        # D["NBand"]=1
        # GD["Beam"]=D
        # BeamMachine=BeamClassLOFARBeam(self.DicoMSInfos["MSName"],GD)
        # BeamMachine.InitBeamMachine()
        # BeamTimes=BM.getBeamSampleTimes()
        # return BM.EstimateBeam(BeamTimes,
        #                        ra,dec)
        


    # def StackAll(self):
    #     while self.iCurrentMS<self.nMS:
    #         self.LoadNextMS()
    #         for iTime in range(self.NTimes):
    #             for iDir in range(self.NDir):
    #                 self.Stack_SingleTimeDir(iTime,iDir)
    #     self.Finalise()

    def StackAll(self):
        while self.iCurrentMS<self.nMS:
            if self.LoadNextMS()=="NotRead": continue
            print("Making dynamic spectra...", file=log)

            FF=self.DicoGrids["DomainEdges_Freq"]
            TT=self.DicoGrids["DomainEdges_Time"]
            for iA0 in range(self.na-1):
                for iA1 in range(iA0+1,self.na):
                    APP.runJob("Stack_SingleTime:%d:%d"%(iA0,iA1), 
                               self.Stack_SingleTime,
                               args=(iA0,iA1))#,serial=True)

            APP.awaitJobResults("Stack_SingleTime:*", progress="Append MS %i"%self.DicoDATA["iMS"])

            
            FOut="%s.Weights.npy"%self.DicoDATA["MSName"]
            
            self.DicoDATA.reload()
            
            # print(self.DicoDATA["WOUT"])
            #self.DicoDATA["WOUT"]/=np.median(self.DicoDATA["WOUT"])
            
            w=self.DicoDATA["WOUT"]

            w/=np.median(w)
            w[w<=0]=0
            wmax=2
            w[w>wmax]=wmax
            
            log.print("    saving weights as %s"%FOut)
            np.save(FOut,self.DicoDATA["WOUT"])
            
            import pylab
            pylab.clf()
            pylab.hist(w[:,0].ravel())
            pylab.draw()
            pylab.show()
            
            #np.save(FOut,self.DicoGrids["GridSTD"])
            
            # for iTime in range(self.NTimes):
            #     self.Stack_SingleTime(iTime)
       
        self.Finalise()


    def killWorkers(self):
        print("Killing workers", file=log)
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()



    def Finalise(self):
        self.killWorkers()
        


    def Stack_SingleTime(self,iA0,iA1):
            

        self.DicoDATA.reload()
        self.DicoGrids.reload()
        DicoMSInfos      = self.DicoMSInfos

        #indRow = np.where((self.DicoDATA["times"]==self.times[iTime]))[0]
        #ch0,ch1=self.DicoGrids["DomainEdges_Freq"][iFreq],self.DicoGrids["DomainEdges_Freq"][iFreq+1]
        nrow,nch,_=self.DicoDATA["data"].shape
        ch0,ch1=0,nch
        
        
        #print(ch0,ch1)
        C0=(self.DicoDATA["A0"]==iA0)&(self.DicoDATA["A1"]==iA1)
        indRow=np.where(C0)[0]
        
        f   = self.DicoDATA["flag"][indRow, ch0:ch1, :]
        d   = self.DicoDATA["data"][indRow, ch0:ch1, :]
        dp   = self.DicoDATA["data_p"][indRow, ch0:ch1, :]
        nrow,nch,_=d.shape
        weights   = (self.DicoDATA["weights"][indRow, ch0:ch1]).reshape((nrow,nch,1))
        A0s = self.DicoDATA["A0"][indRow]
        A1s = self.DicoDATA["A1"][indRow]
        times=self.DicoDATA["times"][indRow]
        NTimesBin=np.unique(times).size
        
        i_TimeBin=np.int64(np.argmin(np.abs(times.reshape((-1,1))-np.sort(np.unique(times)).reshape((1,-1))),axis=1).reshape(-1,1))*np.ones((1,nch))
        i_ch=np.int64((np.arange(nch).reshape(1,-1))*np.ones((nrow,1)))
        i_A0=A0s.reshape((-1,1))*np.ones((1,nch))
        i_A1=A1s.reshape((-1,1))*np.ones((1,nch))

        
        i_TimeBin=np.int64(i_TimeBin).ravel()
        i_ch=np.int64(i_ch).ravel()
        i_A0=np.int64(i_A0).ravel()
        i_A1=np.int64(i_A1).ravel()
        na=self.na
        
        def Give_R(din):
            d0=(din[:,:,0]+din[:,:,-1])/2.
            
            R=np.zeros((nch,NTimesBin),din.dtype)
            
            ind0=i_ch * NTimesBin + i_TimeBin
            
            R.flat[:]=d0.flat[ind0]
            R=d0.T
            
            Rf=np.ones((R.shape),dtype=np.float64)
            Rf[R==0]=0
            
            return R,Rf
        
        iMS  = self.DicoDATA["iMS"]
        
        
        f0, _ = self.Freq_minmax
        

        dcorr=d
        dcorr[f==1]=0
        

        RMS=scipy.stats.median_abs_deviation((dcorr[dcorr!=0]).ravel(),scale="normal")
        df=(dcorr[dcorr!=0]).ravel()
        if df.size>100:
            RMS=np.sqrt(np.abs(np.sum(df*df.conj()))/df.size)
        
        dp[dp==0]=1.
        
        R,Rf=Give_R(dcorr)
        Rp,Rpf=Give_R(dp)
        Rp[Rpf==0]=1.
        #Rp/=np.abs(Rp)
        R/=Rp

        if np.max(np.abs(R))==0: return
        
        C=np.dot(R.T.conj(),R)
        Cn=np.dot(Rf.T,Rf)
        Cn[Cn==0]=1.
        C/=Cn
        
        # import pylab
        # pylab.clf()
        # pylab.subplot(2,2,1)
        # pylab.imshow(np.abs(C),aspect="auto",interpolation="nearest")
        # pylab.colorbar()
        # pylab.title("%i, %i"%(iA0,iA1))
        # pylab.subplot(2,2,2)
        # pylab.imshow(np.abs(R),aspect="auto",interpolation="nearest")
        # pylab.colorbar()
        # pylab.subplot(2,2,3)
        # pylab.imshow(np.abs(Cn),aspect="auto",interpolation="nearest")
        # pylab.colorbar()
        # pylab.title("%i, %i"%(iA0,iA1))
        # pylab.subplot(2,2,4)
        # pylab.imshow(np.abs(Rf),aspect="auto",interpolation="nearest")
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show(block=False)
        # pylab.pause(0.1)
        # return
    
        # ################################

        Cp=np.dot(Rp.T.conj(),Rp)
        Cpn=np.dot(Rpf.T,Rpf)
        Cpn[Cpn==0]=1.
        Cp/=Cpn
        C/=Cp
            

        C=np.abs(C)
        invC=ModLinAlg.invSVD(C[:,:])
    
        w=np.abs(np.sum(invC,axis=0))

        WOUT=self.DicoDATA["WOUT"][indRow,ch0:ch1]
        
        for ii,iRow in enumerate(indRow):
            for ich in np.arange(ch0,ch1):
                self.DicoDATA["WOUT"][iRow,ich]=w[ii]
                
        # for ich in range(nch):
        #     #self.DicoDATA["WOUT"][indRow][:,ich]=w[:]
        #     self.DicoDATA["WOUT"].flat[indRow*nch+ich]=w[:]

    def NormJones(self, G):
        print("  Normalising Jones matrices by the amplitude", file=log)
        G[G != 0.] /= np.abs(G[G != 0.])
        return G
        


    def radec2lm(self, ra, dec):
        # ra and dec must be in radians
        l = np.cos(dec) * np.sin(ra - self.ra0)
        m = np.sin(dec) * np.cos(self.dec0) - np.cos(dec) * np.sin(self.dec0) * np.cos(ra - self.ra0)
        return l, m
# =========================================================================
# =========================================================================
