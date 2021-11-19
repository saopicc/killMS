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
from . import ClassMS
from pyrap.tables import table
from DDFacet.Other import logger
log=logger.getLogger("ClassVisServer")
# import MyPickle
from killMS.Array import NpShared
from killMS.Other import ClassTimeIt
from killMS.Other import ModColor
from killMS.Array import ModLinAlg
logger.setSilent(["NpShared"])
#from Sky.PredictGaussPoints_NumExpr3 import ClassPredictParallel as ClassPredict 
#from Sky.PredictGaussPoints_NumExpr3 import ClassPredict as ClassPredict 
from . import ClassWeighting
from killMS.Other import reformat
import os
from killMS.Other.ModChanEquidistant import IsChanEquidistant
from killMS.Data import ClassReCluster
#import MergeJones
from killMS.Data import ClassJonesDomains
#from DDFacet.Imager import ClassWeighting as ClassWeightingDDF
#from DDFacet.Other.PrintList import ListToStr


class ClassVisServer():
    def __init__(self,MSName,
                 ColName="DATA",
                 TChunkSize=1,
                 TVisSizeMin=1,
                 DicoSelectOptions={},
                 LofarBeam=None,
                 AddNoiseJy=None,IdSharedMem="",
                 SM=None,NCPU=None,
                 Robust=2,Weighting="Natural",
                 WeightUVMinMax=None, WTUV=1.0,
                 GD=None,GDImag=None):

        self.GD=GD
        self.GDImag=GDImag
        self.CalcGridBasedFlags=False
        self.IdSharedMem=IdSharedMem
        PrefixShared="%sSharedVis"%self.IdSharedMem
        self.AddNoiseJy=AddNoiseJy
        self.ReInitChunkCount()
        self.TMemChunkSize=TChunkSize
        self.TVisSizeMin=TVisSizeMin
        self.MSName=MSName
        self.SM=SM
        self.NCPU=NCPU
        self.VisWeights=None
        self.CountPickle=0
        self.ColName=ColName
        self.DicoSelectOptions=DicoSelectOptions
        self.SharedNames=[]
        self.PrefixShared=PrefixShared
        self.VisInSharedMem = (PrefixShared!=None)
        self.LofarBeam=LofarBeam
        self.ApplyBeam=False
        self.DicoClusterDirs_Descriptor=None
        self.Robust=Robust
        self.Weighting=Weighting
        self.DomainsMachine=ClassJonesDomains.ClassJonesDomains()
        self.BeamTimes=np.array([],np.float64)
        self.Init()
        self.dTimesVisMin=self.TVisSizeMin
        self.CurrentVisTimes_SinceStart_Sec=0.,0.
        self.iCurrentVisTime=0
        self.WeightUVMinMax=WeightUVMinMax
        self.WTUV=WTUV
        # self.LoadNextVisChunk()

        # self.TEST_TLIST=[]

    def setSM(self,SM):
        self.SM=SM
        rac,decc=self.MS.radec
        if self.SM.Type=="Catalog":
            self.SM.Calc_LM(rac,decc)

            if self.GD!=None:
                if self.GD["PreApply"]["PreApplySols"][0]!="":
                    CJ=ClassReCluster.ClassReCluster(self.GD)
                    CJ.ReClusterSkyModel(self.SM,self.MS.MSName)
            

    # def SetBeam(self,LofarBeam):
    #     self.BeamMode,self.DtBeamMin,self.BeamRAs,self.BeamDECs = LofarBeam
    #     useArrayFactor=("A" in self.BeamMode)
    #     useElementBeam=("E" in self.BeamMode)
    #     self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
    #     self.ApplyBeam=True
    #     stop

    def Init(self,PointingID=0,NChanJones=1):
        #MSName=self.MDC.giveMS(PointingID).MSName
        kwargs={}
        DecorrMode=""
        ReadUVWDT=False
        if self.GD!=None:
            kwargs["Field"]=self.GD["DataSelection"]["FieldID"]
            kwargs["ChanSlice"]=self.GD["DataSelection"]["ChanSlice"]
            kwargs["DDID"]=self.GD["DataSelection"]["DDID"]
            DecorrMode=self.GD["SkyModel"]["Decorrelation"]
            ReadUVWDT=(("T" in DecorrMode) or ("F" in DecorrMode))
            
        self.ReadUVWDT=ReadUVWDT
        
        ToRaDec=None
        if self.GD is not None and "GDImage" in list(self.GD.keys()):
            ToRaDec=self.GD["GDImage"]["Image"]["PhaseCenterRADEC"]

            if not ToRaDec: # catch empty string
                ToRaDec=None
            
        if ToRaDec=="align":
            log.print(ModColor.Str("kMS does not understand align mode for PhaseCenterRADEC, setting to None..."))
            ToRaDec=None
            #raise RuntimeError("incorrect BeamAt setting: use Facet or Tessel")
        
        MS=ClassMS.ClassMS(self.MSName,Col=self.ColName,DoReadData=False,ReadUVWDT=ReadUVWDT,GD=self.GD,ToRADEC=ToRaDec,**kwargs)
        
        TimesInt=np.arange(0,MS.DTh,self.TMemChunkSize).tolist()
        if not(MS.DTh+1./3600 in TimesInt): TimesInt.append(MS.DTh+1./3600)
        self.TimesInt=TimesInt
        self.NTChunk=len(self.TimesInt)-1
        self.MS=MS
        
        self.DicoMergeStations={}
        
        if self.GD and self.GD["Compression"]["MergeStations"] is not None:
            MergeStations=self.GD["Compression"]["MergeStations"]
            ListMergeNames=[]
            ListMergeStations=[]
            for Name in MergeStations:
                for iAnt in range(MS.na):
                    if Name in MS.StationNames[iAnt]:
                        ListMergeStations.append(iAnt)
                        ListMergeNames.append(MS.StationNames[iAnt])
                        
            log.print("Merging into a single station %s"%str(ListMergeNames))
            self.DicoMergeStations["ListMergeNames"]=ListMergeNames
            self.DicoMergeStations["ListBLMerge"]=[(a0,a1) for a0 in ListMergeStations for a1 in ListMergeStations if a0!=a1]
            self.DicoMergeStations["ListMergeStations"]=ListMergeStations
                        


        ######################################################
        ## Taken from ClassLOFARBeam in DDFacet
        
        ChanWidth=abs(self.MS.ChanWidth.ravel()[0])
        ChanFreqs=self.MS.ChanFreq.flatten()
        if self.GD!=None:
            NChanJones=self.GD["Solvers"]["NChanSols"]
        if NChanJones==0:
            NChanJones=self.MS.NSPWChan
        ChanEdges=np.linspace(ChanFreqs.min()-ChanWidth/2.,ChanFreqs.max()+ChanWidth/2.,NChanJones+1)

        FreqDomains=[[ChanEdges[iF],ChanEdges[iF+1]] for iF in range(NChanJones)]
        FreqDomains=np.array(FreqDomains)
        self.SolsFreqDomains=FreqDomains
        self.NChanJones=NChanJones
        


        MeanFreqJonesChan=(FreqDomains[:,0]+FreqDomains[:,1])/2.
        log.print("Center of frequency domains [MHz]: %s"%str((MeanFreqJonesChan/1e6).tolist()))
        DFreq=np.abs(self.MS.ChanFreq.reshape((self.MS.NSPWChan,1))-MeanFreqJonesChan.reshape((1,NChanJones)))
        self.VisToSolsChanMapping=np.argmin(DFreq,axis=1)
        #log.print(("VisToSolsChanMapping %s"%ListToStr(self.VisToSolsChanMapping)))

        
        self.SolsToVisChanMapping=[]
        for iChanSol in range(NChanJones):
            ind=np.where(self.VisToSolsChanMapping==iChanSol)[0] 
            self.SolsToVisChanMapping.append((ind[0],ind[-1]+1))
        #log.print(("SolsToVisChanMapping %s"%ListToStr(self.SolsToVisChanMapping)))
        

        # ChanDegrid
        FreqBands=ChanEdges
        self.FreqBandsMean=(FreqBands[0:-1]+FreqBands[1::])/2.
        self.FreqBandsMin=FreqBands[0:-1].copy()
        self.FreqBandsMax=FreqBands[1::].copy()
        
        NChanDegrid = NChanJones
        MS=self.MS
        ChanDegridding=np.linspace(FreqBands.min(),FreqBands.max(),NChanDegrid+1)
        FreqChanDegridding=(ChanDegridding[1::]+ChanDegridding[0:-1])/2.
        self.FreqChanDegridding=FreqChanDegridding
        NChanDegrid=FreqChanDegridding.size
        NChanMS=MS.ChanFreq.size
        DChan=np.abs(MS.ChanFreq.reshape((NChanMS,1))-FreqChanDegridding.reshape((1,NChanDegrid)))
        ThisMappingDegrid=np.argmin(DChan,axis=1)
        self.MappingDegrid=ThisMappingDegrid
        #log.print("Mapping degrid: %s"%ListToStr(self.MappingDegrid))
        #NpShared.ToShared("%sMappingDegrid"%self.IdSharedMem,self.MappingDegrid)

        ######################################################
        
        
        # self.CalcWeigths()

        #TimesVisMin=np.arange(0,MS.DTh*60.,self.TVisSizeMin).tolist()
        #if not(MS.DTh*60. in TimesVisMin): TimesVisMin.append(MS.DTh*60.)
        #self.TimesVisMin=np.array(TimesVisMin)




    def CalcWeigths(self,FOV=5.):

        if self.VisWeights!=None: return
        
        uvw,WEIGHT,flags=self.GiveAllUVW()
        u,v,w=uvw.T

        freq=np.mean(self.MS.ChanFreq)
        uvdist=np.sqrt(u**2+v**2)
        uvmax=np.max(uvdist)
        CellSizeRad=res=1./(uvmax*freq/3.e8)
        npix=(FOV*np.pi/180)/res
        
        npix=np.min([npix,30000])

        ImShape=(1,1,npix,npix)
        #VisWeights=WEIGHT[:,0]#np.ones((uvw.shape[0],),dtype=np.float32)
        VisWeights=np.ones((uvw.shape[0],),dtype=np.float32)
        Robust=self.Robust

        # WeightMachine=ClassWeighting.ClassWeighting(ImShape,res)
        # self.VisWeights=WeightMachine.CalcWeights(uvw,VisWeights,Robust=Robust,
        #                                           Weighting=self.Weighting)

        ######################
        #uvw,WEIGHT,flags=self.GiveAllUVW()

        if self.GD is not None:
            if self.GD["Weighting"]["WeightInCol"] is not None and self.GD["Weighting"]["WeightInCol"]!="":
                log.print("Using column %s to compute the weights"%self.GD["Weighting"]["WeightInCol"])
                VisWeights=WEIGHT
            else:
                VisWeights=np.ones((uvw.shape[0],self.MS.ChanFreq.size),dtype=np.float32)
        else:
            VisWeights=np.ones((uvw.shape[0],self.MS.ChanFreq.size),dtype=np.float32)
        
        if np.max(VisWeights)==0.:
            log.print("All imaging weights are 0, setting them to ones")
            VisWeights.fill(1)

        if self.SM.Type=="Image":
            ImShape=self.PaddedFacetShape
            CellSizeRad=self.CellSizeRad

        WeightMachine=ClassWeighting.ClassWeighting(ImShape,CellSizeRad)#res)
        VisWeights=WeightMachine.CalcWeights(uvw,VisWeights,flags,self.MS.ChanFreq,
                                             Robust=Robust,
                                             Weighting=self.Weighting)

        if self.WeightUVMinMax is not None:
            uvmin,uvmax=self.WeightUVMinMax
            log.print('Giving full weight to data in range %f - %f km' % (uvmin, uvmax))
            uvmin*=1000
            uvmax*=1000
            filter=(uvdist<uvmin) | (uvdist>uvmax)
            log.print('Downweighting %i out of %i visibilities' % (np.sum(filter),len(uvdist)))
            VisWeights[filter]*=self.WTUV

        MeanW=np.mean(VisWeights[VisWeights!=0.])
        VisWeights/=MeanW
        log.print('Min weight is %f max is %f' % (np.min(VisWeights),np.max(VisWeights)))
        #VisWeight[VisWeight==0.]=1.
        self.VisWeights=VisWeights

 
    def ReInitChunkCount(self):
        self.CurrentMemTimeChunk=0

    def GiveNextVis(self):

        #log.print( "GiveNextVis")

        t0_bef,t1_bef=self.CurrentVisTimes_SinceStart_Sec
        t0_sec,t1_sec=t1_bef,t1_bef+60.*self.dTimesVisMin

        its_t0,its_t1=self.MS.CurrentChunkTimeRange_SinceT0_sec
        t1_sec=np.min([its_t1,t1_sec])

        self.iCurrentVisTime+=1
        self.CurrentVisTimes_SinceStart_Minutes = t0_sec/60.,t1_sec/60.
        self.CurrentVisTimes_SinceStart_Sec     = t0_sec,t1_sec

        #log.print(("(t0_sec,t1_sec,t1_bef,t1_bef+60.*self.dTimesVisMin)",t0_sec,t1_sec,t1_bef,t1_bef+60.*self.dTimesVisMin))
        #log.print(("(its_t0,its_t1)",its_t0,its_t1))
        #log.print(("self.CurrentVisTimes_SinceStart_Minutes",self.CurrentVisTimes_SinceStart_Minutes))


        #print(self.CurrentVisTimes_SinceStart_Minutes)
        
        if (t0_sec>=its_t1):
            return "EndChunk"

        if not self.have_data:
            return "AllFlaggedThisTime"

        t0_MS=self.MS.F_tstart
        t0_sec+=t0_MS
        t1_sec+=t0_MS
        self.CurrentVisTimes_MS_Sec=t0_sec,t1_sec
        
        D=self.ThisDataChunk

        # Calculate uvw speed for time spearing

        Tmax=self.TimeMemChunkRange_sec_Since70[1]#self.ThisDataChunk["times"][-1]
        # time selection
        indRowsThisChunk=np.where((self.ThisDataChunk["times"]>=t0_sec)&(self.ThisDataChunk["times"]<t1_sec))[0]
        # np.save("indRowsThisChunk.npy",indRowsThisChunk)
        # indRowsThisChunk=np.load("indRowsThisChunk.npy")
        
        if indRowsThisChunk.shape[0]==0:
            if t0_sec>=Tmax:
                return "EndChunk"
            else:
                return "AllFlaggedThisTime"
            
        DATA={}
        DATA["indRowsThisChunk"]=indRowsThisChunk
        for key in D.keys():
            if type(D[key])!=np.ndarray: continue
            if not(key in ['times', 'A1', 'A0', 'flags', 'uvw', 'data', 'Map_VisToJones_Time', "UVW_dt",#"IndexTimesThisChunk", 
                           "W"]):             
                DATA[key]=D[key]
            else:
                DATA[key]=D[key][indRowsThisChunk]

        #############################
        ### data selection
        #############################
        flags=DATA["flags"]
        uvw=DATA["uvw"]
        data=DATA["data"]
        A0=DATA["A0"]
        A1=DATA["A1"]
        times=DATA["times"]
        W=DATA["W"]
        Map_VisToJones_Time=DATA["Map_VisToJones_Time"]
        indRowsThisChunk=DATA["indRowsThisChunk"]
        if self.ReadUVWDT: duvw_dt=DATA["UVW_dt"]

        # IndexTimesThisChunk=DATA["IndexTimesThisChunk"]

        for Field in self.DicoSelectOptions.keys():
            if Field=="UVRangeKm":
                if self.DicoSelectOptions[Field]==None: break
                d0,d1=self.DicoSelectOptions[Field]

                d0*=1e3
                d1*=1e3
                u,v,w=uvw.T
                duv=np.sqrt(u**2+v**2)
                #ind=np.where((duv<d0)|(duv>d1))[0]
                ind=np.where((duv>d0)&(duv<d1))[0]
                
                flags=flags[ind]
                data=data[ind]
                A0=A0[ind]
                A1=A1[ind]
                uvw=uvw[ind]
                times=times[ind]

                #IndexTimesThisChunk=IndexTimesThisChunk[ind]
                W=W[ind]
                Map_VisToJones_Time=Map_VisToJones_Time[ind]
                indRowsThisChunk=indRowsThisChunk[ind]
                if self.ReadUVWDT: duvw_dt=duvw_dt[ind]

        for A in self.FlagAntNumber:
            ind=np.where((A0!=A)&(A1!=A))[0]
            flags=flags[ind]
            data=data[ind]
            A0=A0[ind]
            A1=A1[ind]
            uvw=uvw[ind]
            times=times[ind]
            # IndexTimesThisChunk=IndexTimesThisChunk[ind]
            W=W[ind]
            Map_VisToJones_Time=Map_VisToJones_Time[ind]
            indRowsThisChunk=indRowsThisChunk[ind]
            if self.ReadUVWDT: duvw_dt=duvw_dt[ind]
        
        if self.GD["DataSelection"]["FillFactor"]!=1.:
            Mask=np.random.rand(flags.shape[0])<self.GD["DataSelection"]["FillFactor"]
            ind=np.where(Mask)[0]
            flags=flags[ind]
            data=data[ind]
            A0=A0[ind]
            A1=A1[ind]
            uvw=uvw[ind]
            times=times[ind]
            # IndexTimesThisChunk=IndexTimesThisChunk[ind]
            W=W[ind]
            Map_VisToJones_Time=Map_VisToJones_Time[ind]
            indRowsThisChunk=indRowsThisChunk[ind]
            if self.ReadUVWDT: duvw_dt=duvw_dt[ind]
            
            

        ind=np.where(A0!=A1)[0]
        flags=flags[ind,:,:]
        data=data[ind,:,:]
        A0=A0[ind]
        A1=A1[ind]
        uvw=uvw[ind,:]
        times=times[ind]
        #IndexTimesThisChunk=IndexTimesThisChunk[ind]
        W=W[ind]
        Map_VisToJones_Time=Map_VisToJones_Time[ind]
        indRowsThisChunk=indRowsThisChunk[ind]
        if self.ReadUVWDT: duvw_dt=duvw_dt[ind]
                
        DATA["flags"]=flags
        DATA["rac_decc"]=np.array([self.MS.rac,self.MS.decc])
        DATA["uvw"]=uvw
        DATA["data"]=data
        DATA["A0"]=A0
        DATA["A1"]=A1
        DATA["times"]=times
        #DATA["IndexTimesThisChunk"]=IndexTimesThisChunk
        DATA["W"]=W
        DATA["Map_VisToJones_Time"]=Map_VisToJones_Time
        DATA["indRowsThisChunk"]=indRowsThisChunk
        
        if self.ReadUVWDT: DATA["UVW_dt"]=duvw_dt
                                
        #DATA["UVW_dt"]=self.MS.Give_dUVW_dt(times,A0,A1)
        
        if DATA["flags"].size==0:
            return "AllFlaggedThisTime"
        fFlagged=np.count_nonzero(DATA["flags"])/float(DATA["flags"].size)
        #print fFlagged
        if fFlagged>0.9:
            # log.print( "AllFlaggedThisTime [%f%%]"%(fFlagged*100))
            return "AllFlaggedThisTime"
        
        #if fFlagged==0.:
        #    stop
        # it0=np.min(DATA["IndexTimesThisChunk"])
        # it1=np.max(DATA["IndexTimesThisChunk"])+1
        # DATA["UVW_RefAnt"]=self.ThisDataChunk["UVW_RefAnt"][it0:it1,:,:]
        #
        # # PM=ClassPredict(NCPU=self.NCPU,IdMemShared=self.IdSharedMem)
        # # DATA["Kp"]=PM.GiveKp(DATA,self.SM)

        self.ClearSharedMemory()
        DATA=self.PutInShared(DATA)

        self.SharedVis_Descriptor=NpShared.SharedDicoDescriptor(self.PrefixShared,DATA)

        DATA["A0A1"]=(DATA["A0"],DATA["A1"])

        self.PreApplyJones_Descriptor=None
        if "PreApplyJones" in D.keys():
            NpShared.DicoToShared("%sPreApplyJones"%self.IdSharedMem,D["PreApplyJones"])
            self.PreApplyJones_Descriptor=NpShared.SharedDicoDescriptor("%sPreApplyJones"%self.IdSharedMem,D["PreApplyJones"])


        #it0=np.min(DATA["IndexTimesThisChunk"])
        #it1=np.max(DATA["IndexTimesThisChunk"])+1
        #DATA["UVW_RefAnt"]=self.ThisDataChunk["UVW_RefAnt"][it0:it1,:,:]
        

        #print
        #print self.MS.ROW0,self.MS.ROW1
        #t0=np.min(DATA["times"])-self.MS.F_tstart
        #t1=np.max(DATA["times"])-self.MS.F_tstart
        #self.TEST_TLIST+=sorted(list(set(DATA["times"].tolist())))

        
        return DATA

    def setGridProps(self,Cell,nx):
        self.Cell=Cell
        self.nx=nx
        self.CalcGridBasedFlags=True



    def giveDataSizeAntenna(self):
        t=table(self.MS.MSName,ack=False)
        uvw=t.getcol("UVW")
        flags=t.getcol("FLAG")
        A0,A1=t.getcol("ANTENNA1"),t.getcol("ANTENNA2")
        t.close()
        NVisPerAnt=np.zeros(self.MS.na,np.float64)
        Field="UVRangeKm"
        self.fracNVisPerAnt=np.ones_like(NVisPerAnt)
        NVis=flags[flags==0].size
        if NVis==0:
            log.print( ModColor.Str("Hummm - All the data is flagged!!!"))
            return
        
        if self.DicoSelectOptions[Field] is not None:
            d0,d1=self.DicoSelectOptions[Field]
            
            d0*=1e3
            d1*=1e3
            u,v,w=uvw.T
            duv=np.sqrt(u**2+v**2)
            #ind=np.where((duv<d0)|(duv>d1))[0]
            ind=np.where((duv>d0)&(duv<d1))[0]
            
            flags=flags[ind]
            A0=A0[ind]
            A1=A1[ind]
            uvw=uvw[ind]

            for iAnt in range(self.MS.na):
                NVisPerAnt[iAnt]=np.where((A0==iAnt)|(A1==iAnt))[0].size

            self.fracNVisPerAnt=NVisPerAnt/np.max(NVisPerAnt)
            log.print("Fraction of data per antenna for covariance estimate: %s"%str(self.fracNVisPerAnt.tolist()))


            u,v,w=uvw.T
            d=np.sqrt(u**2+v**2)
            Compactness=np.zeros((self.MS.na,),np.float32)
            for iAnt in range(self.MS.na):
                ind=np.where((A0==iAnt)|(A1==iAnt))[0]
                if ind.size==0:
                    Compactness[iAnt]=1.e-3
                    continue
                Compactness[iAnt]=np.mean(d[ind])
            self.Compactness=Compactness/np.max(Compactness)
            log.print("Compactness: %s"%str(self.Compactness.tolist()))



            
            NVisSel=flags[flags==0].size
            log.print("Total fraction of remaining data after uv-cut: %5.2f %%"%(100*NVisSel/float(NVis)))


    def LoadNextVisChunk(self):
        MS=self.MS

        # bug out when we hit the buffers
        if self.CurrentMemTimeChunk >= self.NTChunk:
            log.print( ModColor.Str("Reached end of observations"))
            self.ReInitChunkCount()
            return "EndOfObservation"

        # get current chunk boundaries
        iT0,iT1=self.CurrentMemTimeChunk,self.CurrentMemTimeChunk+1
        self.CurrentMemTimeChunk+=1

        log.print( "Reading next data chunk in [%5.2f, %5.2f] hours (column %s)"%(self.TimesInt[iT0],self.TimesInt[iT1],MS.ColName))
        self.have_data = MS.ReadData(t0=self.TimesInt[iT0],t1=self.TimesInt[iT1],ReadWeight=True)
        
        if not self.have_data:
            self.CurrentVisTimes_SinceStart_Sec=self.TimesInt[iT0]*3600.,self.TimesInt[iT1]*3600.
            self.CurrentVisTimes_MS_Sec=self.TimesInt[iT0]*3600.+self.MS.F_tstart,self.TimesInt[iT1]*3600.+self.MS.F_tstart
            log.print( ModColor.Str("this data chunk is empty"))
            return "Empty"

        #log.print( "    Rows= [%i, %i]"%(MS.ROW0,MS.ROW1))
        #print float(MS.ROW0)/MS.nbl,float(MS.ROW1)/MS.nbl

        ###############################
        MS=self.MS

        #self.TimeMemChunkRange_sec=MS.times_all[0],MS.times_all[-1]
        self.TimeMemChunkRange_sec=self.TimesInt[iT0]*3600.,self.TimesInt[iT1]*3600.
        self.TimeMemChunkRange_sec_Since70=self.TimesInt[iT0]*3600.+self.MS.F_tstart,self.TimesInt[iT1]*3600.+self.MS.F_tstart
        
        #log.print(("!!!!!!!",self.TimeMemChunkRange_sec))
        times=MS.times_all
        data=MS.data
        A0=MS.A0
        A1=MS.A1
        uvw=MS.uvw
        flags=MS.flag_all
        freqs=MS.ChanFreq.flatten()
        nbl=MS.nbl
        dfreqs=MS.dFreq
        duvw_dt=MS.uvw_dt

        # if Nchan>1:
        #     DoRevertChans=(freqs.flatten()[0]>freqs.flatten()[-1])
        # if self.DoRevertChans:
        #     print ModColor.Str("  ====================== >> Revert Channel order!")
        #     wavelength_chan=wavelength_chan[0,::-1]
        #     freqs=freqs[0,::-1]
        #     self.dFreq=np.abs(self.dFreq)
        
        
        



        #flags.fill(0)

        # f=(np.random.rand(*flags.shape)>0.5)
        # flags[f]=1
        # data[flags]=1e6

        # iAFlag=12
        # ind=np.where((A0==iAFlag)|(A1==iAFlag))[0]
        # flags[ind,:,:]=1
        
        Equidistant=IsChanEquidistant(freqs)
        if freqs.size>1:
            if Equidistant:
                log.print( "Channels are equidistant, can go fast")
            else:
                log.print( ModColor.Str("Channels are not equidistant, cannot go fast"))
        
        MS=self.MS


        if self.SM.Type=="Image":
            u,v,w=uvw.T
            wmax=self.GD["GDImage"]["CF"]["wmax"]
            wmaxkm=wmax/1000.
            log.print( "Flagging baselines with w > %f km"%(wmaxkm))
            C=299792458.
            fmax=self.MS.ChanFreq.ravel()[-1]

            ind=np.where(np.abs(w)>wmax)[0]
            flags[ind,:,:]=1

            f=ind.size/float(flags.shape[0])
            log.print( "  w-Flagged %5.1f%% of the data"%(100*f))



            # data=data[ind]
            # A0=A0[ind]
            # A1=A1[ind]
            # uvw=uvw[ind]
            # times=times[ind]
            # W=W[ind]
            # MapJones=MapJones[ind]
            # indRowsThisChunk=indRowsThisChunk[ind]


        #log.print("::::!!!!!!!!!!!!!!!")
        self.ThresholdFlag=1.#0.9
        self.FlagAntNumber=[]


        ########################################
        
        for A in range(MS.na):
            ind=np.where((MS.A0==A)|(MS.A1==A))[0]
            fA=MS.flag_all[ind].ravel()
            if ind.size==0:
                log.print( "Antenna #%2.2i[%s] is not in the MS"%(A,MS.StationNames[A]))
                self.FlagAntNumber.append(A)
                continue
                
            nf=np.count_nonzero(fA)
            
            Frac=nf/float(fA.size)
            if Frac>self.ThresholdFlag:
                log.print( "Taking antenna #%2.2i[%s] out of the solve (~%4.1f%% of flagged data, more than %4.1f%%)"%\
                    (A,MS.StationNames[A],Frac*100,self.ThresholdFlag*100))
                self.FlagAntNumber.append(A)
                
        if self.CalcGridBasedFlags:
            Cell=self.Cell
            nx=self.nx
            MS=self.MS
            u,v,w=MS.uvw.T
            d=np.sqrt(u**2+v**2)
            CellRad=(Cell/3600.)*np.pi/180
            #_,_,nx,ny=GridShape

            # ###
            # S=CellRad*nx
            # C=3e8
            # freqs=MS.ChanFreq
            # x=d.reshape((d.size,1))*(freqs.reshape((1,freqs.size))/C)*S
            # fA_all=(x>(nx/2))
            # ###
            
            C=3e8
            freqs=MS.ChanFreq.flatten()
            x=d.reshape((d.size,1))*(freqs.reshape((1,freqs.size))/C)*CellRad
            fA_all=(x>(1./2))
            
            for A in range(MS.na):
                ind=np.where((MS.A0==A)|(MS.A1==A))[0]
                fA=fA_all[ind].ravel()
                nf=np.count_nonzero(fA)
                if fA.size==0:
                    Frac=1.0
                else:
                    Frac=nf/float(fA.size)
                if Frac>self.ThresholdFlag:
                    log.print( "Taking antenna #%2.2i[%s] out of the solve (~%4.1f%% of out-grid data, more than %4.1f%%)"%\
                               (A,MS.StationNames[A],Frac*100,self.ThresholdFlag*100))
                    self.FlagAntNumber.append(A)




        if "FlagAnts" in self.DicoSelectOptions.keys():
            FlagAnts=self.DicoSelectOptions["FlagAnts"]
            for Name in FlagAnts:
                for iAnt in range(MS.na):
                    if Name in MS.StationNames[iAnt]:
                        log.print( "Taking antenna #%2.2i[%s] out of the solve"%(iAnt,MS.StationNames[iAnt]))
                        self.FlagAntNumber.append(iAnt)



                        
        if "DistMaxToCore" in self.DicoSelectOptions.keys():
            DMax=self.DicoSelectOptions["DistMaxToCore"]*1e3
            X,Y,Z=MS.StationPos.T
            Xm,Ym,Zm=np.median(MS.StationPos,axis=0).flatten().tolist()
            D=np.sqrt((X-Xm)**2+(Y-Ym)**2+(Z-Zm)**2)
            ind=np.where(D>DMax)[0]

            for iAnt in ind.tolist():
                log.print("Taking antenna #%2.2i[%s] out of the solve (distance to core: %.1f km)"%(iAnt,MS.StationNames[iAnt],D[iAnt]/1e3))
                self.FlagAntNumber.append(iAnt)
            



        #############################
        #############################

        ind=np.where(np.isnan(data))
        flags[ind]=1
        
        # ## debug
        # ind=np.where((A0==0)&(A1==1))[0]
        # flags=flags[ind]
        # data=data[ind]
        # A0=A0[ind]
        # A1=A1[ind]
        # uvw=uvw[ind]
        # times=times[ind]
        # ##


        if self.AddNoiseJy!=None:
            data+=(self.AddNoiseJy/np.sqrt(2.))*(np.random.randn(*data.shape)+1j*np.random.randn(*data.shape))
            stop
        # Building uvw infos
        #################################################
        #################################################
        # log.print( "Building uvw infos .... ")
        # Luvw=np.zeros((MS.times.size,MS.na,3),uvw.dtype)
        # AntRef=0
        # indexTimes=np.zeros((times.size,),np.int64)
        # iTime=0

        # # UVW per antenna
        # #Times_all_32=np.float32(times-MS.times[0])
        # #Times32=np.float32(MS.times-MS.times[0])
        # irow=0
        # for ThisTime in MS.times:#Times32:

        #     T= ClassTimeIt.ClassTimeIt("VS")
        #     T.disable()
        #     #ind=np.where(Times_all_32[irow::]==ThisTime)[0]
        #     ind=np.where(times[irow::]==ThisTime)[0]+irow
            
        #     irow+=ind.size
        #     T.timeit("0b")

        #     indAnt=np.where(A0[ind]==AntRef)[0]
        #     ThisUVW0=uvw[ind][indAnt].copy()
        #     Ant0=A1[ind][indAnt].copy()
        #     T.timeit("1")

        #     indAnt=np.where(A1[ind]==AntRef)[0]
        #     ThisUVW1=-uvw[ind][indAnt].copy()
        #     Ant1=A0[ind][indAnt].copy()
        #     ThisUVW=np.concatenate((ThisUVW1,ThisUVW0[1::]))

        #     T.timeit("2")

        #     #AA=np.concatenate((Ant1,Ant0[1::]))
        #     Luvw[iTime,:,:]=ThisUVW[:,:]
        
        #     T.timeit("3")

        #     #Luvw.append(ThisUVW)
        #     indexTimes[ind]=iTime
        #     iTime+=1
        # log.print( "     .... Done ")
        #################################################
        #################################################


        # Dt_UVW_dt=1.*3600
        # t0=times[0]
        # t1=times[-1]
        # All_UVW_dt=[]
        # Times=np.arange(t0,t1,Dt_UVW_dt).tolist()
        # if not(t1 in Times): Times.append(t1+1)

        # All_UVW_dt=np.array([],np.float32).reshape((0,3))
        # for it in range(len(Times)-1):
        #     t0=Times[it]
        #     t1=Times[it+1]
        #     tt=(t0+t1)/2.
        #     indRows=np.where((times>=t0)&(times<t1))[0]
        #     All_UVW_dt=np.concatenate((All_UVW_dt,self.MS.Give_dUVW_dt(tt,A0[indRows],A1[indRows])))

            

        # UVW_dt=All_UVW_dt



        PredictedData=np.zeros_like(data)
        Indices=np.arange(PredictedData.size).reshape(PredictedData.shape)
        NpShared.ToShared("%sPredictedData"%self.IdSharedMem,PredictedData)
        NpShared.ToShared("%sIndicesData"%self.IdSharedMem,Indices)
        
        PredictedDataGains=np.zeros_like(data)
        IndicesGains=np.arange(PredictedDataGains.size).reshape(PredictedDataGains.shape)
        NpShared.ToShared("%sPredictedDataGains"%self.IdSharedMem,PredictedDataGains)
        NpShared.ToShared("%sIndicesDataGains"%self.IdSharedMem,IndicesGains)
        
        
        
        #NpShared.PackListArray("%sUVW_Ants"%self.IdSharedMem,Luvw)
        #self.UVW_RefAnt=NpShared.ToShared("%sUVW_RefAnt"%self.IdSharedMem,Luvw)
        #self.IndexTimes=NpShared.ToShared("%sIndexTimes"%self.IdSharedMem,indexTimes)
        ThisDataChunk={"times":times,
                       "freqs":freqs,
                       "dfreqs":dfreqs,
                       "freqs_full":freqs,
                       "dfreqs_full":dfreqs,
                       #"A0A1":(A0[ind],A1[ind]),
                       #"A0A1":(A0,A1),
                       "A0":A0,
                       "A1":A1,
                       "uvw":uvw,
                       "flags":flags,
                       "nbl":nbl,
                       "na":MS.na,
                       "data":data,
                       "ROW0":MS.ROW0,
                       "ROW1":MS.ROW1,
                       "infos":np.array([MS.na,MS.TimeInterVal[0]]),
                       #"IndexTimesThisChunk":indexTimes,
                       #"UVW_RefAnt": Luvw,
                       "W":self.VisWeights[MS.ROW0:MS.ROW1],
                       #"IndRows_All_UVW_dt":IndRows_All_UVW_dt,
                       "UVW_dt":duvw_dt
                     }


        self.ThisDataChunk=ThisDataChunk#NpShared.DicoToShared("%sThisDataChunk"%self.IdSharedMem,ThisDataChunk)
        #self.UpdateCompression()
        #self.ThisDataChunk["Map_VisToJones_Time"]=np.zeros(([],),np.int32)
        self.ThisDataChunk["Map_VisToJones_Time"]=np.zeros((times.size,),np.int32)


        ListDicoPreApply=[]
        DoPreApplyJones=False
        if self.GD is not None:
            if self.GD["Beam"]["BeamModel"] is not None:
                
                if self.GD["Beam"]["BeamAt"].lower() == "tessel":
                    log.print("Estimating Beam directions at the center of the tesselated areas")
                    RA,DEC=self.SM.ClusterCat.ra,self.SM.ClusterCat.dec
                elif self.GD["Beam"]["BeamAt"].lower() == "facet":
                    log.print("Estimating Beam directions at the center of the individual facets areas")
                    RA=np.array([self.SM.DicoImager[iFacet]["RaDec"][0] for iFacet in range(len(self.SM.DicoImager))])
                    DEC=np.array([self.SM.DicoImager[iFacet]["RaDec"][1] for iFacet in range(len(self.SM.DicoImager))])
                else:
                    raise RuntimeError("incorrect BeamAt setting: use Facet or Tessel")
                
                if self.GD["Beam"]["BeamModel"]=="LOFAR":
                    NDir=RA.size
                    self.DtBeamMin=self.GD["Beam"]["DtBeamMin"]
                    useArrayFactor=("A" in self.GD["Beam"]["LOFARBeamMode"])
                    useElementBeam=("E" in self.GD["Beam"]["LOFARBeamMode"])
                    self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
                    log.print( "Update LOFAR beam in %i directions [Dt = %3.1f min] ... "%(NDir,self.DtBeamMin))
                    DtBeamSec=self.DtBeamMin*60
                    tmin,tmax=np.min(times)-MS.dt/2.,np.max(times)+MS.dt/2.
                    # TimesBeam=np.arange(np.min(times),np.max(times),DtBeamSec).tolist()
                    # if not(tmax in TimesBeam): TimesBeam.append(tmax)
                    NTimesBeam=round((tmax-tmin)/DtBeamSec)
                    NTimesBeam=int(np.max([2,NTimesBeam]))


                    TimesBeam=np.linspace(np.min(times)-1,np.max(times)+1,NTimesBeam).tolist()
                    TimesBeam=np.array(TimesBeam)

                    T0s=TimesBeam[:-1]
                    T1s=TimesBeam[1:]
                    Tm=(T0s+T1s)/2.
                    
                    self.BeamTimes=TimesBeam
                    # print "!!!!!!!!!!!!!!!!!!!!"
                    # T0s=MS.F_times-MS.dt/2.
                    # T1s=MS.F_times+MS.dt/2.
                    # Tm=MS.F_times

                    # from killMS.Other.rad2hmsdms import rad2hmsdms
                    # for i in range(RA.size): 
                    #     ra,dec=RA[i],DEC[i]
                    #     print rad2hmsdms(ra,Type="ra").replace(" ",":"),rad2hmsdms(dec,Type="dec").replace(" ",".")

                    Beam=np.zeros((Tm.size,NDir,self.MS.na,self.MS.NSPWChan,2,2),np.complex64)
                    for itime in range(Tm.size):
                        ThisTime=Tm[itime]
                        Beam[itime]=self.MS.GiveBeam(ThisTime,RA,DEC)
    
                    # # Beam[:,76,:,:,0,0]=20.
                    # # Beam[:,76,:,:,0,1]=0.
                    # # Beam[:,76,:,:,1,0]=0.
                    # # Beam[:,76,:,:,1,1]=20.
                    # Beam[:,:,:,:,0,0]=20.
                    # Beam[:,:,:,:,0,1]=0.
                    # Beam[:,:,:,:,1,0]=0.
                    # Beam[:,:,:,:,1,1]=20.


                    DicoBeam={}
                    DicoBeam["t0"]=T0s
                    DicoBeam["t1"]=T1s
                    DicoBeam["tm"]=Tm
                    DicoBeam["Jones"]=Beam

                    ChanWidth=self.MS.ChanWidth.ravel()[0]
                    ChanFreqs=self.MS.ChanFreq.flatten()

                    self.DomainsMachine.AddFreqDomains(DicoBeam,ChanFreqs,ChanWidth)
                    
                    NChanBeam=self.GD["Beam"]["NChanBeamPerMS"]
                    if NChanBeam==0:
                        NChanBeam=self.MS.NSPWChan
                    FreqDomainsOut=self.DomainsMachine.GiveFreqDomains(ChanFreqs,ChanWidth,NChanJones=NChanBeam)
                    self.DomainsMachine.AverageInFreq(DicoBeam,FreqDomainsOut)

                    ###### Normalise
                    rac,decc=self.MS.OriginalRadec
                    if self.GD["Beam"]["CenterNorm"]==1:

                        Beam=DicoBeam["Jones"]
                        Beam0=np.zeros((Tm.size,1,self.MS.na,self.MS.NSPWChan,2,2),np.complex64)
                        for itime in range(Tm.size):
                            ThisTime=Tm[itime]
                            Beam0[itime]=self.MS.GiveBeam(ThisTime,np.array([rac]),np.array([decc]))

                            
                        DicoBeamCenter={}
                        DicoBeamCenter["t0"]=T0s
                        DicoBeamCenter["t1"]=T1s
                        DicoBeamCenter["tm"]=Tm
                        DicoBeamCenter["Jones"]=Beam0
                        self.DomainsMachine.AddFreqDomains(DicoBeamCenter,ChanFreqs,ChanWidth)
                        self.DomainsMachine.AverageInFreq(DicoBeamCenter,FreqDomainsOut)
                        Beam0=DicoBeamCenter["Jones"]
                        Beam0inv=ModLinAlg.BatchInverse(Beam0)
                        nt,nd,_,_,_,_=Beam.shape
                        Ones=np.ones((nt,nd, 1, 1, 1, 1),np.float32)
                        Beam0inv=Beam0inv*Ones
                        DicoBeam["Jones"]=ModLinAlg.BatchDot(Beam0inv,Beam)

                    ###### 




                    #nt,nd,na,nch,_,_= Beam.shape
                    #Beam=np.mean(Beam,axis=3).reshape((nt,nd,na,1,2,2))
                    
                    #DicoBeam["ChanMap"]=np.zeros((nch))
                    ListDicoPreApply.append(DicoBeam)

                    DoPreApplyJones=True
                    log.print( "       .... done Update LOFAR beam ")
                elif self.GD["Beam"]["BeamModel"] == "FITS" or self.GD["Beam"]["BeamModel"] == "ATCA":
                    NDir = RA.size
                    self.DtBeamMin = self.GD["Beam"]["DtBeamMin"]

                    if self.GD["Beam"]["BeamModel"] == "FITS":
                        from DDFacet.Data.ClassFITSBeam import ClassFITSBeam as ClassDDFBeam
                    elif self.GD["Beam"]["BeamModel"] == "ATCA":
                        from DDFacet.Data.ClassATCABeam import ClassATCABeam as ClassDDFBeam
                        
                    # make fake opts dict (DDFacet clss expects slightly different option names)
                    opts = self.GD["Beam"]
                    opts["NBand"] = opts["NChanBeamPerMS"]
                    ddfbeam = ClassDDFBeam(self.MS, opts)

                    TimesBeam = np.array(ddfbeam.getBeamSampleTimes(times))
                    FreqDomains = ddfbeam.getFreqDomains()
                    nfreq_dom = FreqDomains.shape[0]

                    log.print( "Update %s beam in %i dirs, %i times, %i freqs ... " % (self.GD["Beam"]["BeamModel"],NDir, len(TimesBeam), nfreq_dom))

                    T0s = TimesBeam[:-1]
                    T1s = TimesBeam[1:]
                    Tm = (T0s + T1s) / 2.

                    self.BeamTimes = TimesBeam

                    Beam = np.zeros((Tm.size, NDir, self.MS.na, FreqDomains.shape[0], 2, 2), np.complex64)
                    for itime, tm in enumerate(Tm):
                        Beam[itime] = ddfbeam.evaluateBeam(tm, RA, DEC)

                    DicoBeam = {}
                    DicoBeam["t0"] = T0s
                    DicoBeam["t1"] = T1s
                    DicoBeam["tm"] = Tm
                    DicoBeam["Jones"] = Beam
                    DicoBeam["FreqDomain"] = FreqDomains

                    ###### Normalise
                    #rac, decc = self.MS.radec
                    rac,decc=self.MS.OriginalRadec
                    if self.GD["Beam"]["CenterNorm"] == 1:

                        Beam = DicoBeam["Jones"]
                        Beam0 = np.zeros((Tm.size, 1, self.MS.na, nfreq_dom, 2, 2), np.complex64)
                        for itime, tm in enumerate(Tm):
                            Beam0[itime] = ddfbeam.evaluateBeam(tm, np.array([rac]), np.array([decc]))

                        DicoBeamCenter = {}
                        DicoBeamCenter["t0"] = T0s
                        DicoBeamCenter["t1"] = T1s
                        DicoBeamCenter["tm"] = Tm
                        DicoBeamCenter["Jones"] = Beam0
                        DicoBeamCenter["FreqDomain"] = FreqDomains
                        Beam0inv = ModLinAlg.BatchInverse(Beam0)
                        nt, nd, _, _, _, _ = Beam.shape
                        Ones = np.ones((nt, nd, 1, 1, 1, 1), np.float32)
                        Beam0inv = Beam0inv * Ones
                        DicoBeam["Jones"] = ModLinAlg.BatchDot(Beam0inv, Beam)

                    ######




                    # nt,nd,na,nch,_,_= Beam.shape
                    # Beam=np.mean(Beam,axis=3).reshape((nt,nd,na,1,2,2))

                    # DicoBeam["ChanMap"]=np.zeros((nch))
                    ListDicoPreApply.append(DicoBeam)

                    DoPreApplyJones = True
                    log.print( "       .... done Update beam ")
                elif self.GD["Beam"]["BeamModel"] == "GMRT":
                    NDir = RA.size
                    self.DtBeamMin = self.GD["Beam"]["DtBeamMin"]

                    from DDFacet.Data.ClassGMRTBeam import ClassGMRTBeam
                    # make fake opts dict (DDFacet clss expects slightly different option names)
                    opts = self.GD["Beam"]
                    opts["NBand"] = opts["NChanBeamPerMS"]
                    gmrtbeam = ClassGMRTBeam(self.MS, opts)

                    TimesBeam = np.array(gmrtbeam.getBeamSampleTimes(times))
                    FreqDomains = gmrtbeam.getFreqDomains()
                    nfreq_dom = FreqDomains.shape[0]

                    log.print( "Update GMRT beam in %i dirs, %i times, %i freqs ... " % (NDir, len(TimesBeam), nfreq_dom))

                    T0s = TimesBeam[:-1]
                    T1s = TimesBeam[1:]
                    Tm = (T0s + T1s) / 2.

                    self.BeamTimes = TimesBeam

                    Beam = np.zeros((Tm.size, NDir, self.MS.na, FreqDomains.shape[0], 2, 2), np.complex64)
                    for itime, tm in enumerate(Tm):
                        Beam[itime] = gmrtbeam.GiveInstrumentBeam(tm, RA, DEC)

                    DicoBeam = {}
                    DicoBeam["t0"] = T0s
                    DicoBeam["t1"] = T1s
                    DicoBeam["tm"] = Tm
                    DicoBeam["Jones"] = Beam
                    DicoBeam["FreqDomain"] = FreqDomains

                    ###### Normalise
                    #rac, decc = self.MS.radec
                    rac,decc=self.MS.OriginalRadec
                    if self.GD["Beam"]["CenterNorm"] == 1:

                        Beam = DicoBeam["Jones"]
                        Beam0 = np.zeros((Tm.size, 1, self.MS.na, nfreq_dom, 2, 2), np.complex64)
                        for itime, tm in enumerate(Tm):
                            Beam0[itime] = gmrtbeam.evaluateBeam(tm, np.array([rac]), np.array([decc]))

                        DicoBeamCenter = {}
                        DicoBeamCenter["t0"] = T0s
                        DicoBeamCenter["t1"] = T1s
                        DicoBeamCenter["tm"] = Tm
                        DicoBeamCenter["Jones"] = Beam0
                        DicoBeamCenter["FreqDomain"] = FreqDomains
                        Beam0inv = ModLinAlg.BatchInverse(Beam0)
                        nt, nd, _, _, _, _ = Beam.shape
                        Ones = np.ones((nt, nd, 1, 1, 1, 1), np.float32)
                        Beam0inv = Beam0inv * Ones
                        DicoBeam["Jones"] = ModLinAlg.BatchDot(Beam0inv, Beam)

                    ######




                    # nt,nd,na,nch,_,_= Beam.shape
                    # Beam=np.mean(Beam,axis=3).reshape((nt,nd,na,1,2,2))

                    # DicoBeam["ChanMap"]=np.zeros((nch))
                    ListDicoPreApply.append(DicoBeam)

                    DoPreApplyJones = True
                    log.print( "       .... done Update GMRT beam ")

            if self.GD["PreApply"]["PreApplySols"][0]!="":
                ModeList=self.GD["PreApply"]["PreApplyMode"]
                if ModeList==[""]: ModeList=["AP"]*len(self.GD["PreApply"]["PreApplySols"])
                for SolFile,Mode in zip(self.GD["PreApply"]["PreApplySols"],ModeList):
                    log.print( "Loading solution file %s in %s mode"%(SolFile,Mode))

                    if (SolFile!="")&(not(".npz" in SolFile)):
                        Method=SolFile
                        ThisMSName=reformat.reformat(os.path.abspath(self.MS.MSName),LastSlash=False)
                        SolFileLoad="%s/killMS.%s.sols.npz"%(ThisMSName,Method)
                        if self.GD["Solutions"]["SolsDir"]:
                            _MSName=reformat.reformat(self.MSName).split("/")[-2]
                            DirName="%s%s"%(reformat.reformat(self.GD["Solutions"]["SolsDir"]),_MSName)
                            SolFileLoad="%s/killMS.%s.sols.npz"%(DirName,SolFile)
                    else:
                        SolFileLoad=SolFile

                    S=np.load(SolFileLoad)
                    Sols=S["Sols"]
                    nt,nch,na,nd,_,_=Sols["G"].shape
                    
                    DicoSols={}
                    DicoSols["t0"]=Sols["t0"]
                    DicoSols["t1"]=Sols["t1"]
                    DicoSols["tm"]=(Sols["t0"]+Sols["t1"])/2.
                    DicoSols["Jones"]=np.swapaxes(Sols["G"],1,3).reshape((nt,nd,na,nch,2,2))
                    DicoSols["FreqDomain"]=S["FreqDomains"]
                    if not("A" in Mode):
                        ind=(DicoSols["Jones"]!=0.)
                        DicoSols["Jones"][ind]/=np.abs(DicoSols["Jones"][ind])
                    if not("P" in Mode):
                        dtype=DicoSols["Jones"].dtype
                        DicoSols["Jones"]=(np.abs(DicoSols["Jones"]).astype(dtype)).copy()

                    #DicoSols["Jones"]=Sols["G"].reshape((nt,nd,na,1,2,2))
                    ListDicoPreApply.append(DicoSols)
                    DoPreApplyJones=True


                    
            
        if DoPreApplyJones:
            DicoJones=ListDicoPreApply[0]
            DomainsMachine=self.DomainsMachine

            if self.SM.Type=="Image":
                DomainsMachine.setFacetToDirMapping([self.SM.DicoImager[iFacet]["iDirJones"] for iFacet in range(len(self.SM.DicoImager))])
            for DicoJones1 in ListDicoPreApply[1::]:
                DicoJones=DomainsMachine.MergeJones(DicoJones1,DicoJones)
            
            DomainsMachine.AddVisToJonesMapping(DicoJones,self.ThisDataChunk["times"],self.ThisDataChunk["freqs"])
            
            
            # ind=np.zeros((times.size,),np.int32)
            # #nt,na,nd,_,_,_=Beam.shape
            # ii=0
            # for it in range(nt):
            #     t0=DicoJones["t0"][it]
            #     t1=DicoJones["t1"][it]
            #     indMStime=np.where((times>=t0)&(times<t1))[0]
            #     indMStime=np.ones((indMStime.size,),np.int32)*it
            #     ind[ii:ii+indMStime.size]=indMStime[:]
            #     ii+=indMStime.size
            # TimeMapping=ind

            #DicoJones["ChanMap"]=self.VisToJonesChanMapping
            #self.ThisDataChunk["MapJones"]=TimeMapping



            self.ThisDataChunk["PreApplyJones"]=DicoJones

            DicoClusterDirs={}
            DicoClusterDirs["l"]=self.SM.ClusterCat.l
            DicoClusterDirs["m"]=self.SM.ClusterCat.m
            DicoClusterDirs["ra"]=self.SM.ClusterCat.ra
            DicoClusterDirs["dec"]=self.SM.ClusterCat.dec
            DicoClusterDirs["I"]=self.SM.ClusterCat.SumI
            DicoClusterDirs["Cluster"]=self.SM.ClusterCat.Cluster
            
            NpShared.DicoToShared("%sDicoClusterDirs"%self.IdSharedMem,DicoClusterDirs)
            self.DicoClusterDirs_Descriptor=NpShared.SharedDicoDescriptor("%sDicoClusterDirs"%self.IdSharedMem,DicoClusterDirs)

            self.ThisDataChunk["Map_VisToJones_Time"]=self.ThisDataChunk["PreApplyJones"]["Map_VisToJones_Time"]

        return "LoadOK"


    def setFOV(self,FullImShape,PaddedFacetShape,FacetShape,CellSizeRad):
        self.FullImShape=FullImShape
        self.PaddedFacetShape=PaddedFacetShape
        self.FacetShape=FacetShape
        self.CellSizeRad=CellSizeRad

    def UpdateCompression(self):
        ThisMSName=reformat.reformat(os.path.abspath(self.MS.MSName),LastSlash=False)

        D=self.ThisDataChunk

        DATA={}
        DATA["flags"]=D["flags"]
        DATA["data"]=D["data"]
        DATA["uvw"]=D["uvw"]
        DATA["A0"]=D["A0"]
        DATA["A1"]=D["A1"]
        DATA["times"]=D["times"]

        GD=self.GD["GDImage"]
        if GD["Compression"]["CompDeGridMode"]:
            MapName="%s/Mapping.CompDeGrid.npy"%ThisMSName
            try:
                FinalMapping=np.load(MapName)
            except:
                if GD["Compression"]["CompDeGridFOV"]=="Facet":
                    _,_,nx,ny=self.FacetShape
                elif GD["Compression"]["CompDeGridFOV"]=="Full":
                    _,_,nx,ny=self.FullImShape
                FOV=self.CellSizeRad*nx*(np.sqrt(2.)/2.)*180./np.pi
                SmearMapMachine=ClassSmearMapping.ClassSmearMapping(self.MS,radiusDeg=FOV,Decorr=(1.-GD["Compression"]["CompDeGridDecorr"]),IdSharedMem=self.IdSharedMem,NCPU=self.NCPU)
                #SmearMapMachine.BuildSmearMapping(DATA)
                FinalMapping,fact=SmearMapMachine.BuildSmearMappingParallel(DATA)
                np.save(MapName,FinalMapping)
                log.print( ModColor.Str("  Effective compression [DeGrid]:   %.2f%%"%fact,col="green"))

            Map=NpShared.ToShared("%sMappingSmearing.DeGrid"%(self.IdSharedMem),FinalMapping)


    def GiveAllUVW(self):
        t=self.MS.GiveMainTable()
        uvw=t.getcol("UVW")
        if self.GD is not None:
            if self.GD["Weighting"]["WeightInCol"] is not None and self.GD["Weighting"]["WeightInCol"]!="":
                WEIGHT=t.getcol(self.GD["Weighting"]["WeightInCol"])
            else:
                WEIGHT=t.getcol("WEIGHT")
        else:
            WEIGHT=t.getcol("WEIGHT")

        F=t.getcol("FLAG")
        t.close()

        return uvw,WEIGHT,F


    def ClearSharedMemory(self):
        NpShared.DelAll(self.PrefixShared)
        NpShared.DelAll("%sDicoData"%self.IdSharedMem)
        NpShared.DelAll("%sKernelMat"%self.IdSharedMem)

        # for Name in self.SharedNames:
        #     NpShared.DelArray(Name)
        self.SharedNames=[]

    def PutInShared(self,Dico):
        DicoOut={}
        for key in Dico.keys():
            if type(Dico[key])!=np.ndarray: continue
            #print "%s.%s"%(self.PrefixShared,key)
            Shared=NpShared.ToShared("%s.%s"%(self.PrefixShared,key),Dico[key])
            DicoOut[key]=Shared
            self.SharedNames.append("%s.%s"%(self.PrefixShared,key))
            
        return DicoOut

