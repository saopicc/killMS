import numpy as np
import ClassMS
from pyrap.tables import table
import MyLogger
log=MyLogger.getLogger("ClassVisServer")
# import MyPickle
import NpShared
import ClassTimeIt
import ModColor

class ClassVisServer():
    def __init__(self,MSName,
                 ColName="DATA",
                 TChunkSize=1,
                 PrefixShared="Default",
                 DicoSelectOptions={}):
  
        self.ReInitChunkCount()
        self.TChunkSize=TChunkSize
        self.MSName=MSName
        self.VisWeights=None
        self.CountPickle=0
        self.ColName=ColName
        self.DicoSelectOptions=DicoSelectOptions
        self.SharedNames=[]
        self.PrefixShared=None
        self.VisInSharedMem = (PrefixShared!=None)

        self.Init()

    def Init(self,PointingID=0):
        #MSName=self.MDC.giveMS(PointingID).MSName
        MS=ClassMS.ClassMS(self.MSName,Col=self.ColName,DoReadData=False)
        TimesInt=np.arange(0,MS.DTh,self.TChunkSize).tolist()
        if not(MS.DTh in TimesInt): TimesInt.append(MS.DTh)
        self.TimesInt=TimesInt
        self.NTChunk=len(self.TimesInt)-1
        self.MS=MS

    def ReInitChunkCount(self):
        self.CurrentTimeChunk=0

    def GiveNextVis(self,t0_sec,t1_sec):
        t0,t1=t0_sec,t1_sec


        if self.MS.CurrentTimeHoursRange!=None:
            its_t0,its_t1=self.MS.CurrentTimeHoursRange
            if not((t0>=its_t0)&(t1<=its_t1)):
                self.LoadNextVisChunk()
        else:
            self.LoadNextVisChunk()

        t=self.ThisDataChunk["times"]
        t0_MS=self.MS.F_tstart
        t0+=t0_MS
        t1+=t0_MS

        ind=np.where((t>=t0)&(t<t1))[0]
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



        return DATA



    def LoadNextVisChunk(self):
        if self.CurrentTimeChunk==self.NTChunk:
            print>>log, "Reached end of chunks"
            self.ReInitChunkCount()
            return None
        MS=self.MS
        iT0,iT1=self.CurrentTimeChunk,self.CurrentTimeChunk+1
        self.CurrentTimeChunk+=1

        print>>log, "Reading next data chunk in [%5.2f, %5.2f] hours"%(self.TimesInt[iT0],self.TimesInt[iT1])
        MS.ReadData(t0=self.TimesInt[iT0],t1=self.TimesInt[iT1])
        #print>>log, "    Rows= [%i, %i]"%(MS.ROW0,MS.ROW1)
        #print float(MS.ROW0)/MS.nbl,float(MS.ROW1)/MS.nbl

        ###############################
        MS=self.MS
        it0=0
        it1=-1
        row0=it0*MS.nbl
        row1=it1*MS.nbl
        if it1==-1:
            row1=None

        times=MS.times_all[row0:row1]
        data=MS.data[row0:row1]
        A0=MS.A0[row0:row1]
        A1=MS.A1[row0:row1]
        uvw=MS.uvw[row0:row1]
        flags=MS.flag_all[row0:row1]
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

        flags=flags[ind]
        data=data[ind]
        A0=A0[ind]
        A1=A1[ind]
        uvw=uvw[ind]
        times=times[ind]

        # ## debug
        # ind=np.where((A0==0)&(A1==1))[0]
        # flags=flags[ind]
        # data=data[ind]
        # A0=A0[ind]
        # A1=A1[ind]
        # uvw=uvw[ind]
        # times=times[ind]
        # ##



        
        DicoDataOut={"itimes":(it0,it1),
                     "times":times,#[ind],
                     "freqs":freqs,
                     #"A0A1":(A0[ind],A1[ind]),
                     "A0A1":(A0,A1),
                     "uvw":uvw,#[ind],
                     "flags":flags,#[ind],
                     "nbl":nbl,
                     "na":MS.na,
                     "data":data,
                     "ROW0":MS.ROW0,
                     "ROW1":MS.ROW1,
                     "infos":np.array([MS.na])
                     }
        

        #MyPickle.Save(DicoDataOut,"Pickle_All_%2.2i"%self.CountPickle)
        #self.CountPickle+=1

        DATA=DicoDataOut

        A0,A1=DATA["A0A1"]
        DATA["A0"]=A0
        DATA["A1"]=A1

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
        for Name in self.SharedNames:
            NpShared.DelArray(Name)
        self.SharedNames=[]

    def PutInShared(self,Dico):
        print>>log, ModColor.Str("Sharing data: start [prefix = %s]"%self.PrefixShared)
        DicoOut={}
        for key in Dico.keys():
            if type(Dico[key])!=np.ndarray: continue
            Shared=NpShared.ToShared("%s.%s"%(self.PrefixShared,key),Dico[key])
            DicoOut[key]=Shared
            self.SharedNames.append("%s.%s"%(self.PrefixShared,key))
        print>>log, ModColor.Str("Sharing data: done")
        return DicoOut

