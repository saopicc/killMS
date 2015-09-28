import numpy as np
import ClassMS
from pyrap.tables import table
from killMS2.Other import MyLogger
log=MyLogger.getLogger("ClassVisServer")
# import MyPickle
from killMS2.Array import NpShared
from killMS2.Other import ClassTimeIt
from killMS2.Other import ModColor
from killMS2.Array import ModLinAlg
MyLogger.setSilent(["NpShared"])
#from Sky.PredictGaussPoints_NumExpr3 import ClassPredictParallel as ClassPredict 
#from Sky.PredictGaussPoints_NumExpr3 import ClassPredict as ClassPredict 
import ClassWeighting
from killMS2.Other import reformat
import os
from killMS2.Other.ModChanEquidistant import IsChanEquidistant
from killMS2.Data import ClassJones
import MergeJones

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
        
        self.Robust=Robust
        self.Weighting=Weighting

        self.Init()

        self.dTimesVisMin=self.TVisSizeMin
        self.CurrentVisTimes_SinceStart_Sec=0.,0.
        self.iCurrentVisTime=0

        # self.LoadNextVisChunk()

        # self.TEST_TLIST=[]

    def setSM(self,SM):
        self.SM=SM
        rac,decc=self.MS.radec
        self.SM.Calc_LM(rac,decc)
        if self.GD["PreApply"]["PreApplySols"][0]!="":
            CJ=ClassJones.ClassJones(self.GD)
            CJ.ReClusterSkyModel(self.SM,self.MS.MSName)
            

    # def SetBeam(self,LofarBeam):
    #     self.BeamMode,self.DtBeamMin,self.BeamRAs,self.BeamDECs = LofarBeam
    #     useArrayFactor=("A" in self.BeamMode)
    #     useElementBeam=("E" in self.BeamMode)
    #     self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
    #     self.ApplyBeam=True
    #     stop

    def Init(self,PointingID=0):
        #MSName=self.MDC.giveMS(PointingID).MSName
        MS=ClassMS.ClassMS(self.MSName,Col=self.ColName,DoReadData=False)

        TimesInt=np.arange(0,MS.DTh,self.TMemChunkSize).tolist()
        if not(MS.DTh in TimesInt): TimesInt.append(MS.DTh)
        self.TimesInt=TimesInt
        self.NTChunk=len(self.TimesInt)-1
        self.MS=MS
        self.CalcWeigths()

        #TimesVisMin=np.arange(0,MS.DTh*60.,self.TVisSizeMin).tolist()
        #if not(MS.DTh*60. in TimesVisMin): TimesVisMin.append(MS.DTh*60.)
        #self.TimesVisMin=np.array(TimesVisMin)

    def CalcWeigths(self,FOV=5.):
        if self.VisWeights!=None: return
        
        uvw,WEIGHT=self.GiveAllUVW()
        u,v,w=uvw.T
        freq=np.mean(self.MS.ChanFreq)
        uvmax=np.max(np.sqrt(u**2+v**2))
        res=uvmax*freq/3.e8
        npix=(FOV*np.pi/180)/res
        ImShape=(1,1,npix,npix)
        WeightMachine=ClassWeighting.ClassWeighting(ImShape,res)
        #VisWeights=WEIGHT[:,0]#np.ones((uvw.shape[0],),dtype=np.float32)
        VisWeights=np.ones((uvw.shape[0],),dtype=np.float32)
        Robust=self.Robust
        self.VisWeights=WeightMachine.CalcWeights(uvw,VisWeights,Robust=Robust,
                                                  Weighting=self.Weighting)
 
    def ReInitChunkCount(self):
        self.CurrentMemTimeChunk=0

    def GiveNextVis(self):

        #print>>log, "GiveNextVis"

        t0_bef,t1_bef=self.CurrentVisTimes_SinceStart_Sec
        t0_sec,t1_sec=t1_bef,t1_bef+60.*self.dTimesVisMin

        its_t0,its_t1=self.MS.CurrentChunkTimeRange_SinceT0_sec
        t1_sec=np.min([its_t1,t1_sec])

        self.iCurrentVisTime+=1
        self.CurrentVisTimes_SinceStart_Minutes = t0_sec/60.,t1_sec/60.
        self.CurrentVisTimes_SinceStart_Sec     = t0_sec,t1_sec

        #print>>log,("(t0_sec,t1_sec,t1_bef,t1_bef+60.*self.dTimesVisMin)",t0_sec,t1_sec,t1_bef,t1_bef+60.*self.dTimesVisMin)
        #print>>log,("(its_t0,its_t1)",its_t0,its_t1)
        #print>>log,("self.CurrentVisTimes_SinceStart_Minutes",self.CurrentVisTimes_SinceStart_Minutes)

        if (t0_sec>=its_t1):
            return "EndChunk"

        

        
        t0_MS=self.MS.F_tstart
        t0_sec+=t0_MS
        t1_sec+=t0_MS
        self.CurrentVisTimes_MS_Sec=t0_sec,t1_sec
        
        D=self.ThisDataChunk

        # Calculate uvw speed for time spearing


        # time selection
        ind=np.where((self.ThisDataChunk["times"]>=t0_sec)&(self.ThisDataChunk["times"]<t1_sec))[0]
        if ind.shape[0]==0:
            return "EndChunk"
        DATA={}
        for key in D.keys():
            if type(D[key])!=np.ndarray: continue
            if not(key in ['times', 'A1', 'A0', 'flags', 'uvw', 'data', 'MapJones', #"IndexTimesThisChunk", 
                           "W"]):             
                DATA[key]=D[key]
            else:
                DATA[key]=D[key][ind]

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
        MapJones=DATA["MapJones"]
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
                MapJones=MapJones[ind]


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
            MapJones=MapJones[ind]
        
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
            MapJones=MapJones[ind]
            
            

        ind=np.where(A0!=A1)[0]
        flags=flags[ind,:,:]
        data=data[ind,:,:]
        A0=A0[ind]
        A1=A1[ind]
        uvw=uvw[ind,:]
        times=times[ind]
        #IndexTimesThisChunk=IndexTimesThisChunk[ind]
        W=W[ind]
        MapJones=MapJones[ind]

        DATA["flags"]=flags
        DATA["uvw"]=uvw
        DATA["data"]=data
        DATA["A0"]=A0
        DATA["A1"]=A1
        DATA["times"]=times
        #DATA["IndexTimesThisChunk"]=IndexTimesThisChunk
        DATA["W"]=W
        DATA["MapJones"]=MapJones

        DATA["UVW_dt"]=self.MS.Give_dUVW_dt(times,A0,A1)
        


        # it0=np.min(DATA["IndexTimesThisChunk"])
        # it1=np.max(DATA["IndexTimesThisChunk"])+1
        # DATA["UVW_RefAnt"]=self.ThisDataChunk["UVW_RefAnt"][it0:it1,:,:]
        #
        # # PM=ClassPredict(NCPU=self.NCPU,IdMemShared=self.IdSharedMem)
        # # DATA["Kp"]=PM.GiveKp(DATA,self.SM)

        #stop


        if self.VisInSharedMem:
            self.ClearSharedMemory()
            DATA=self.PutInShared(DATA)
            DATA["A0A1"]=(DATA["A0"],DATA["A1"])


        if "PreApplyJones" in D.keys():
            NpShared.DicoToShared("%sPreApplyJones"%self.IdSharedMem,D["PreApplyJones"])

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





    def LoadNextVisChunk(self):
        if self.CurrentMemTimeChunk==self.NTChunk:
            print>>log, ModColor.Str("Reached end of observations")
            self.ReInitChunkCount()
            return "EndOfObservation"
        MS=self.MS
        iT0,iT1=self.CurrentMemTimeChunk,self.CurrentMemTimeChunk+1
        self.CurrentMemTimeChunk+=1

        print>>log, "Reading next data chunk in [%5.2f, %5.2f] hours"%(self.TimesInt[iT0],self.TimesInt[iT1])
        MS.ReadData(t0=self.TimesInt[iT0],t1=self.TimesInt[iT1],ReadWeight=True)




        
        #print>>log, "    Rows= [%i, %i]"%(MS.ROW0,MS.ROW1)
        #print float(MS.ROW0)/MS.nbl,float(MS.ROW1)/MS.nbl

        ###############################
        MS=self.MS

        self.TimeMemChunkRange_sec=MS.times_all[0],MS.times_all[-1]

        times=MS.times_all
        data=MS.data
        A0=MS.A0
        A1=MS.A1
        uvw=MS.uvw
        flags=MS.flag_all
        freqs=MS.ChanFreq.flatten()
        nbl=MS.nbl
        dfreqs=MS.dFreq
        #flags.fill(0)

        # f=(np.random.rand(*flags.shape)>0.5)
        # flags[f]=1
        # data[flags]=1e6

        # iAFlag=12
        # ind=np.where((A0==iAFlag)|(A1==iAFlag))[0]
        # flags[ind,:,:]=1
        
        Equidistant=IsChanEquidistant(freqs)
        if Equidistant:
            print>>log, "Channels are equidistant, can go fast"
        else:
            print>>log, ModColor.Str("Channels are not equidistant, cannot go fast")

        MS=self.MS
        self.ThresholdFlag=0.9
        self.FlagAntNumber=[]
        for A in range(MS.na):
            ind=np.where((MS.A0==A)|(MS.A1==A))[0]
            fA=MS.flag_all[ind].ravel()
            nf=np.count_nonzero(fA)
            Frac=nf/float(fA.size)
            if Frac>self.ThresholdFlag:
                print>>log, "Taking antenna #%2.2i[%s] out of the solve (~%4.1f%% of flagged data, more than %4.1f%%)"%\
                    (A,MS.StationNames[A],Frac*100,self.ThresholdFlag*100)
                self.FlagAntNumber.append(A)
                
        if self.CalcGridBasedFlags:
            Cell=self.Cell
            nx=self.nx
            MS=self.MS
            u,v,w=MS.uvw.T
            d=np.sqrt(u**2+v**2)
            CellRad=(Cell/3600.)*np.pi/180
            #_,_,nx,ny=GridShape
            S=CellRad*nx
            C=3e8
            freqs=MS.ChanFreq
            x=d.reshape((d.size,1))*(freqs.reshape((1,freqs.size))/C)*S
            fA_all=(x>(nx/2))
            
            for A in range(MS.na):
                ind=np.where((MS.A0==A)|(MS.A1==A))[0]
                fA=fA_all[ind].ravel()
                nf=np.count_nonzero(fA)
                Frac=nf/float(fA.size)
                if Frac>self.ThresholdFlag:
                    print>>log, "Taking antenna #%2.2i[%s] out of the solve (~%4.1f%% of out-grid data, more than %4.1f%%)"%\
                        (A,MS.StationNames[A],Frac*100,self.ThresholdFlag*100)
                    self.FlagAntNumber.append(A)



        if "FlagAnts" in self.DicoSelectOptions.keys():
            FlagAnts=self.DicoSelectOptions["FlagAnts"]
            for Name in FlagAnts:
                for iAnt in range(MS.na):
                    if Name in MS.StationNames[iAnt]:
                        print>>log, "Taking antenna #%2.2i[%s] out of the solve"%(iAnt,MS.StationNames[iAnt])
                        self.FlagAntNumber.append(iAnt)
        if "DistMaxToCore" in self.DicoSelectOptions.keys():
            DMax=self.DicoSelectOptions["DistMaxToCore"]*1e3
            X,Y,Z=MS.StationPos.T
            Xm,Ym,Zm=np.median(MS.StationPos,axis=0).flatten().tolist()
            D=np.sqrt((X-Xm)**2+(Y-Ym)**2+(Z-Zm)**2)
            ind=np.where(D>DMax)[0]

            for iAnt in ind.tolist():
                print>>log,"Taking antenna #%2.2i[%s] out of the solve (distance to core: %.1f km)"%(iAnt,MS.StationNames[iAnt],D[iAnt]/1e3)
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
        # print>>log, "Building uvw infos .... "
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
        # print>>log, "     .... Done "
        #################################################
        #################################################


        Dt_UVW_dt=1.*3600
        t0=times[0]
        t1=times[-1]
        All_UVW_dt=[]
        Times=np.arange(t0,t1,Dt_UVW_dt).tolist()
        if not(t1 in Times): Times.append(t1+1)

        All_UVW_dt=np.array([],np.float32).reshape((0,3))
        for it in range(len(Times)-1):
            t0=Times[it]
            t1=Times[it+1]
            tt=(t0+t1)/2.
            indRows=np.where((times>=t0)&(times<t1))[0]
            All_UVW_dt=np.concatenate((All_UVW_dt,self.MS.Give_dUVW_dt(tt,A0[indRows],A1[indRows])))

            

        UVW_dt=All_UVW_dt
        


        #NpShared.PackListArray("%sUVW_Ants"%self.IdSharedMem,Luvw)
        #self.UVW_RefAnt=NpShared.ToShared("%sUVW_RefAnt"%self.IdSharedMem,Luvw)
        #self.IndexTimes=NpShared.ToShared("%sIndexTimes"%self.IdSharedMem,indexTimes)
        ThisDataChunk={"times":times,
                       "freqs":freqs,
                       "dfreqs":dfreqs,
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
                       "UVW_dt":UVW_dt
                     }
        
        self.ThisDataChunk=ThisDataChunk#NpShared.DicoToShared("%sThisDataChunk"%self.IdSharedMem,ThisDataChunk)
        #self.UpdateCompression()
        self.ThisDataChunk["MapJones"]=np.zeros((times.size,),np.int32)


        ListDicoPreApply=[]
        DoPreApplyJones=False
        if self.GD!=None:
            if self.GD["Beam"]["BeamModel"]!=None:
                if self.GD["Beam"]["BeamModel"]=="LOFAR":
                    self.DtBeamMin=self.GD["Beam"]["DtBeamMin"]
                    useArrayFactor=("A" in self.GD["Beam"]["LOFARBeamMode"])
                    useElementBeam=("E" in self.GD["Beam"]["LOFARBeamMode"])
                    self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
                    print>>log, "Update LOFAR beam [Dt = %3.1f min] ... "%self.DtBeamMin
                    DtBeamSec=self.DtBeamMin*60
                    tmin,tmax=np.min(times),np.max(times)
                    TimesBeam=np.arange(np.min(times),np.max(times),DtBeamSec).tolist()
                    if not(tmax in TimesBeam): TimesBeam.append(tmax)
                    TimesBeam=np.array(TimesBeam)
                    T0s=TimesBeam[:-1]
                    T1s=TimesBeam[1:]
                    Tm=(T0s+T1s)/2.
                    RA,DEC=self.SM.ClusterCat.ra,self.SM.ClusterCat.dec
                    NDir=RA.size
                    Beam=np.zeros((Tm.size,NDir,self.MS.na,self.MS.NSPWChan,2,2),np.complex64)
                    for itime in range(Tm.size):
                        ThisTime=Tm[itime]
                        Beam[itime]=self.MS.GiveBeam(ThisTime,RA,DEC)
    
                    ###### Normalise
                    rac,decc=self.MS.radec
                    if self.GD["Beam"]["CenterNorm"]==1:
                        for itime in range(Tm.size):
                            ThisTime=Tm[itime]
                            Beam0=self.MS.GiveBeam(ThisTime,np.array([rac]),np.array([decc]))
                            Beam0inv=ModLinAlg.BatchInverse(Beam0)
                            nd,_,_,_,_=Beam[itime].shape
                            Ones=np.ones((nd, 1, 1, 1, 1),np.float32)
                            Beam0inv=Beam0inv*Ones
                            Beam[itime]=ModLinAlg.BatchDot(Beam0inv,Beam[itime])

                    ###### 



                    nt,nd,na,nch,_,_= Beam.shape
                    Beam=np.mean(Beam,axis=3).reshape((nt,nd,na,1,2,2))
                    
    
                    DicoBeam={}
                    DicoBeam["t0"]=T0s
                    DicoBeam["t1"]=T1s
                    DicoBeam["tm"]=Tm
                    DicoBeam["Jones"]=Beam
                    ListDicoPreApply.append(DicoBeam)

                    DoPreApplyJones=True
                    print>>log, "       .... done Update LOFAR beam "

        
            if self.GD["PreApply"]["PreApplySols"][0]!="":
                ModeList=self.GD["PreApply"]["PreApplyMode"]
                if ModeList==[""]: ModeList=["AP"]*len(self.GD["PreApply"]["PreApplySols"])
                for SolFile,Mode in zip(self.GD["PreApply"]["PreApplySols"],ModeList):
                    print>>log, "Loading solution file %s in %s mode"%(SolFile,Mode)

                    if (SolFile!="")&(not(".npz" in SolFile)):
                        Method=SolFile
                        ThisMSName=reformat.reformat(os.path.abspath(self.MS.MSName),LastSlash=False)
                        SolFileLoad="%s/killMS.%s.sols.npz"%(ThisMSName,Method)
                    else:
                        SolFileLoad=SolFile

                    Sols=np.load(SolFileLoad)["Sols"]
                    nt,na,nd,_,_=Sols["G"].shape
                    DicoSols={}
                    DicoSols["t0"]=Sols["t0"]
                    DicoSols["t1"]=Sols["t1"]
                    DicoSols["tm"]=(Sols["t0"]+Sols["t1"])/2.
                    DicoSols["Jones"]=np.swapaxes(Sols["G"],1,2).reshape((nt,nd,na,1,2,2))
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

            for DicoJones1 in ListDicoPreApply[1::]:
                DicoJones=MergeJones.MergeJones(DicoJones1,DicoJones)
                

            ind=np.zeros((times.size,),np.int32)
            #nt,na,nd,_,_,_=Beam.shape
            ii=0
            for it in range(nt):
                t0=DicoJones["t0"][it]
                t1=DicoJones["t1"][it]
                indMStime=np.where((times>=t0)&(times<t1))[0]
                indMStime=np.ones((indMStime.size,),np.int32)*it
                ind[ii:ii+indMStime.size]=indMStime[:]
                ii+=indMStime.size
            TimeMapping=ind

            self.ThisDataChunk["MapJones"]=TimeMapping
            self.ThisDataChunk["PreApplyJones"]=DicoJones
            
            DicoClusterDirs={}
            DicoClusterDirs["l"]=self.SM.ClusterCat.l
            DicoClusterDirs["m"]=self.SM.ClusterCat.m
            DicoClusterDirs["ra"]=self.SM.ClusterCat.ra
            DicoClusterDirs["dec"]=self.SM.ClusterCat.dec
            DicoClusterDirs["I"]=self.SM.ClusterCat.SumI
            DicoClusterDirs["Cluster"]=self.SM.ClusterCat.Cluster
            
            NpShared.DicoToShared("%sDicoClusterDirs"%self.IdSharedMem,DicoClusterDirs)
            

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
                print>>log, ModColor.Str("  Effective compression [DeGrid]:   %.2f%%"%fact,col="green")

            Map=NpShared.ToShared("%sMappingSmearing.DeGrid"%(self.IdSharedMem),FinalMapping)


    def GiveAllUVW(self):
        t=table(self.MS.MSName,ack=False)
        uvw=t.getcol("UVW")
        WEIGHT=t.getcol("WEIGHT")
        t.close()
        return uvw,WEIGHT


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

