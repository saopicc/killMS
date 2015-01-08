import numpy as np
import ClassMS
from pyrap.tables import table
import MyLogger
log=MyLogger.getLogger("ClassVisServer")
# import MyPickle
import NpShared
import ClassTimeIt
import ModColor
import ModLinAlg

class ClassVisServer():
    def __init__(self,MSName,
                 ColName="DATA",
                 TChunkSize=1,
                 TVisSizeMin=1,
                 PrefixShared="SharedVis",
                 DicoSelectOptions={},
                 LofarBeam=None):
  
        self.ReInitChunkCount()
        self.TMemChunkSize=TChunkSize
        self.TVisSizeMin=TVisSizeMin
        self.MSName=MSName
        self.VisWeights=None
        self.CountPickle=0
        self.ColName=ColName
        self.DicoSelectOptions=DicoSelectOptions
        self.SharedNames=[]
        self.PrefixShared=PrefixShared
        self.VisInSharedMem = (PrefixShared!=None)
        self.LofarBeam=LofarBeam
        self.ApplyBeam=False
        self.Init()

    def SetBeam(self,LofarBeam):
        self.BeamMode,self.DtBeamMin,self.BeamRAs,self.BeamDECs = LofarBeam
        useArrayFactor=("A" in self.BeamMode)
        useElementBeam=("E" in self.BeamMode)
        self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
        self.ApplyBeam=True

    def Init(self,PointingID=0):
        #MSName=self.MDC.giveMS(PointingID).MSName
        MS=ClassMS.ClassMS(self.MSName,Col=self.ColName,DoReadData=False)
        TimesInt=np.arange(0,MS.DTh,self.TMemChunkSize).tolist()
        if not(MS.DTh in TimesInt): TimesInt.append(MS.DTh)
        self.TimesInt=TimesInt
        self.NTChunk=len(self.TimesInt)-1
        self.MS=MS

        TimesVisMin=np.arange(0,MS.DTh*60.,self.TVisSizeMin).tolist()
        if not(MS.DTh*60. in TimesVisMin): TimesVisMin.append(MS.DTh*60.)
        self.TimesVisMin=np.array(TimesVisMin)
        self.iCurrentVisTime=0

    def ReInitChunkCount(self):
        self.CurrentMemTimeChunk=0

    def GiveNextVis(self,t0_sec=None,t1_sec=None):

        if t0_sec==None:
            if self.iCurrentVisTime+1==self.TimesVisMin.size: return None
            t0_sec,t1_sec=60.*self.TimesVisMin[self.iCurrentVisTime],60.*self.TimesVisMin[self.iCurrentVisTime+1]
            self.iCurrentVisTime+=1

        t0,t1=t0_sec,t1_sec
        self.CurrentVisTimes=t0_sec,t1_sec
        if self.MS.CurrentChunkTimeRange_SinceT0_sec!=None:
            its_t0,its_t1=self.MS.CurrentChunkTimeRange_SinceT0_sec
            if not((t0>=its_t0)&(t1<=its_t1)):
                self.LoadNextVisChunk()
        else:
            self.LoadNextVisChunk()
        
        t0_MS=self.MS.F_tstart
        t0+=t0_MS
        t1+=t0_MS

        ind=np.where((self.ThisDataChunk["times"]>=t0)&(self.ThisDataChunk["times"]<t1))[0]
        D=self.ThisDataChunk



        DATA={}
        for key in D.keys():
            if type(D[key])!=np.ndarray: continue
            if not(key in ['times', 'A1', 'A0', 'flags', 'uvw', 'data']):             
                DATA[key]=D[key]
            else:
                DATA[key]=D[key][ind]

        if self.VisInSharedMem:
            self.ClearSharedMemory()
            DATA=self.PutInShared(DATA)
            DATA["A0A1"]=(DATA["A0"],DATA["A1"])

        if "DicoBeam" in D.keys():
            DATA["DicoBeam"]=D["DicoBeam"]

        #print
        #print self.MS.ROW0,self.MS.ROW1
        #t0=np.min(DATA["times"])-self.MS.F_tstart
        #t1=np.max(DATA["times"])-self.MS.F_tstart
        #self.TEST_TLIST+=sorted(list(set(DATA["times"].tolist())))

        return DATA




    def LoadNextVisChunk(self):
        if self.CurrentMemTimeChunk==self.NTChunk:
            print>>log, "Reached end of chunks"
            self.ReInitChunkCount()
            return None
        MS=self.MS
        iT0,iT1=self.CurrentMemTimeChunk,self.CurrentMemTimeChunk+1
        self.CurrentMemTimeChunk+=1

        print>>log, "Reading next data chunk in [%5.2f, %5.2f] hours"%(self.TimesInt[iT0],self.TimesInt[iT1])
        MS.ReadData(t0=self.TimesInt[iT0],t1=self.TimesInt[iT1])




        
        #print>>log, "    Rows= [%i, %i]"%(MS.ROW0,MS.ROW1)
        #print float(MS.ROW0)/MS.nbl,float(MS.ROW1)/MS.nbl

        ###############################
        MS=self.MS

        times=MS.times_all
        data=MS.data
        A0=MS.A0
        A1=MS.A1
        uvw=MS.uvw
        flags=MS.flag_all
        freqs=MS.ChanFreq.flatten()
        nbl=MS.nbl





        for Field in self.DicoSelectOptions.keys():
            if Field=="UVRangeKm":
                d0,d1=Field
                d0*=1e3
                d1*=1e3
                u,v,w=MS.uvw.T
                duv=np.sqrt(u**2+v**2)
                ind=np.where((duv<d0)|(duv>d1))[0]
                
                flags=flags[ind]
                data=data[ind]
                A0=A0[ind]
                A1=A1[ind]
                uvw=uvw[ind]
                times=times[ind]


        ind=np.where(A0!=A1)[0]
        flags=flags[ind,:,:].copy()
        data=data[ind,:,:].copy()
        A0=A0[ind].copy()
        A1=A1[ind].copy()
        uvw=uvw[ind,:].copy()
        times=times[ind].copy()

        # ## debug
        # ind=np.where((A0==0)&(A1==1))[0]
        # flags=flags[ind]
        # data=data[ind]
        # A0=A0[ind]
        # A1=A1[ind]
        # uvw=uvw[ind]
        # times=times[ind]
        # ##



        
        DicoDataOut={"times":times,
                     "freqs":freqs,
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
                     "infos":np.array([MS.na])
                     }
        

        if self.ApplyBeam:
            print>>log, "Update LOFAR beam .... "
            DtBeamSec=self.DtBeamMin*60
            tmin,tmax=np.min(times),np.max(times)
            TimesBeam=np.arange(np.min(times),np.max(times),DtBeamSec).tolist()
            if not(tmax in TimesBeam): TimesBeam.append(tmax)
            TimesBeam=np.array(TimesBeam)
            T0s=TimesBeam[:-1]
            T1s=TimesBeam[1:]
            Tm=(T0s+T1s)/2.
            RA,DEC=self.BeamRAs,self.BeamDECs
            NDir=RA.size
            Beam=np.zeros((Tm.size,NDir,self.MS.na,self.MS.NSPWChan,2,2),np.complex64)
            for itime in range(Tm.size):
                ThisTime=Tm[itime]
                Beam[itime]=self.MS.GiveBeam(ThisTime,self.BeamRAs,self.BeamDECs)
            BeamH=ModLinAlg.BatchH(Beam)

            DicoBeam={}
            DicoBeam["t0"]=T0s
            DicoBeam["t1"]=T1s
            DicoBeam["tm"]=Tm
            DicoBeam["Beam"]=Beam
            DicoBeam["BeamH"]=BeamH
            DicoDataOut["DicoBeam"]=DicoBeam
            
            print>>log, "       .... done Update LOFAR beam "

        #MyPickle.Save(DicoDataOut,"Pickle_All_%2.2i"%self.CountPickle)
        #self.CountPickle+=1

        DATA=DicoDataOut

        #A0,A1=DATA["A0A1"]
        #DATA["A0"]=A0
        #DATA["A1"]=A1

        ##############################################
        
        
        # DATA["data"].fill(1)

        # DATA.keys()
        # ['uvw', 'MapBLSel', 'Weights', 'nbl', 'data', 'ROW_01', 'itimes', 'freqs', 'nf', 'times', 'A1', 'A0', 'flags', 'nt', 'A0A1']

        self.ThisDataChunk=DATA



    def GiveAllUVW(self):
        t=table(self.MS.MSName,ack=False)
        uvw=t.getcol("UVW")
        t.close()
        return uvw


    def ClearSharedMemory(self):
        NpShared.DelAll()
        # for Name in self.SharedNames:
        #     NpShared.DelArray(Name)
        self.SharedNames=[]

    def PutInShared(self,Dico):
        print>>log, ModColor.Str("Sharing data: start [prefix = %s]"%self.PrefixShared)
        DicoOut={}
        for key in Dico.keys():
            if type(Dico[key])!=np.ndarray: continue
            print "%s.%s"%(self.PrefixShared,key)
            Shared=NpShared.ToShared("%s.%s"%(self.PrefixShared,key),Dico[key])
            DicoOut[key]=Shared
            self.SharedNames.append("%s.%s"%(self.PrefixShared,key))
        print>>log, ModColor.Str("Sharing data: done")
        return DicoOut

