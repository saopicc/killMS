import numpy as np
from pyrap.tables import table
from Data.ClassMS import ClassMS
from Sky.ClassSM import ClassSM
from Other.ClassTimeIt import ClassTimeIt
import numexpr as ne
#import ModNumExpr
from Other.progressbar import ProgressBar
import multiprocessing
from Array import ModLinAlg
from Array import NpShared
#ne.evaluate=lambda sin: ("return %s"%sin)
import time
from Predict import predict 
from Other import findrms

def SolsToDicoJones(Sols,nf):
    Jones={}
    Jones["t0"]=Sols.t0
    Jones["t1"]=Sols.t1
    nt,na,nd,_,_=Sols.G.shape
    G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))
    Jones["Beam"]=G
    Jones["BeamH"]=ModLinAlg.BatchH(G)
    Jones["ChanMap"]=np.zeros((nf,))#.tolist()
    return Jones


class ClassPredictParallel():
    def __init__(self,Precision="S",NCPU=6,IdMemShared="",DoSmearing=False):
        self.NCPU=NCPU
        ne.set_num_threads(self.NCPU)
        if Precision=="D":
            self.CType=np.complex128
            self.FType=np.float64
        if Precision=="S":
            self.CType=np.complex64
            self.FType=np.float32
        self.IdMemShared=IdMemShared
        self.DoSmearing=DoSmearing
        self.PM=ClassPredict(Precision=Precision,NCPU=NCPU,IdMemShared=IdMemShared,DoSmearing=DoSmearing)





    def GiveCovariance(self,DicoDataIn,ApplyTimeJones,SM):

        DicoData=NpShared.DicoToShared("%sDicoMemChunk"%(self.IdMemShared),DicoDataIn,DelInput=False)
        
        if ApplyTimeJones!=None:
            ApplyTimeJones=NpShared.DicoToShared("%sApplyTimeJones"%(self.IdMemShared),ApplyTimeJones,DelInput=False)

        nrow,nch,_=DicoData["data"].shape
        
        RowList=np.int64(np.linspace(0,nrow,self.NCPU+1))
        row0=RowList[0:-1]
        row1=RowList[1::]
        
        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        workerlist=[]
        NCPU=self.NCPU
        
        # NpShared.DelArray("%sCorrectedData"%(self.IdMemShared))
        # CorrectedData=NpShared.SharedArray.create("%sCorrectedData"%(self.IdMemShared),DicoData["data"].shape,dtype=DicoData["data"].dtype)
        # CorrectedData=

        for ii in range(NCPU):
            W=WorkerPredict(work_queue, result_queue,self.IdMemShared,Mode="GiveCovariance",DoSmearing=self.DoSmearing,SM=SM)
            workerlist.append(W)
            workerlist[ii].start()

        NJobs=row0.size
        for iJob in range(NJobs):
            ThisJob=row0[iJob],row1[iJob]
            work_queue.put(ThisJob)

        while int(result_queue.qsize())<NJobs:
            time.sleep(0.1)
            continue
 
        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

        DicoDataIn["W"]=DicoData["W"]


    def ApplyCal(self,DicoDataIn,ApplyTimeJones,iCluster):

        DicoData=NpShared.DicoToShared("%sDicoMemChunk"%(self.IdMemShared),DicoDataIn,DelInput=False)
        if ApplyTimeJones!=None:
            ApplyTimeJones=NpShared.DicoToShared("%sApplyTimeJones"%(self.IdMemShared),ApplyTimeJones,DelInput=False)

        nrow,nch,_=DicoData["data"].shape
        
        RowList=np.int64(np.linspace(0,nrow,self.NCPU+1))
        row0=RowList[0:-1]
        row1=RowList[1::]
        
        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        workerlist=[]
        NCPU=self.NCPU
        
        # NpShared.DelArray("%sCorrectedData"%(self.IdMemShared))
        # CorrectedData=NpShared.SharedArray.create("%sCorrectedData"%(self.IdMemShared),DicoData["data"].shape,dtype=DicoData["data"].dtype)
        # CorrectedData=

        for ii in range(NCPU):
            W=WorkerPredict(work_queue, result_queue,self.IdMemShared,Mode="ApplyCal",iCluster=iCluster)
            workerlist.append(W)
            workerlist[ii].start()

        NJobs=row0.size
        for iJob in range(NJobs):
            ThisJob=row0[iJob],row1[iJob]
            work_queue.put(ThisJob)

        while int(result_queue.qsize())<NJobs:
            time.sleep(0.1)
            continue
 
        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

        DicoDataIn["data"]=DicoData["data"]


    def predictKernelPolCluster(self,DicoData,SM,iDirection=None,ApplyJones=None,ApplyTimeJones=None,Noise=None):

        DicoData=NpShared.DicoToShared("%sDicoMemChunk"%(self.IdMemShared),DicoData,DelInput=False)
        if ApplyTimeJones!=None:
            ApplyTimeJones=NpShared.DicoToShared("%sApplyTimeJones"%(self.IdMemShared),ApplyTimeJones,DelInput=False)

        nrow,nch,_=DicoData["data"].shape
        
        RowList=np.int64(np.linspace(0,nrow,self.NCPU+1))
        row0=RowList[0:-1]
        row1=RowList[1::]
        
        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        workerlist=[]
        NCPU=self.NCPU
        
        NpShared.DelArray("%sPredictData"%(self.IdMemShared))
        PredictArray=NpShared.SharedArray.create("%sPredictData"%(self.IdMemShared),DicoData["data"].shape,dtype=DicoData["data"].dtype)
        
        for ii in range(NCPU):
            W=WorkerPredict(work_queue, result_queue,self.IdMemShared,SM=SM,DoSmearing=self.DoSmearing)
            workerlist.append(W)
            workerlist[ii].start()

        NJobs=row0.size
        for iJob in range(NJobs):
            ThisJob=row0[iJob],row1[iJob]
            work_queue.put(ThisJob)

        while int(result_queue.qsize())<NJobs:
            time.sleep(0.1)
            continue
 
        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()
            
        return PredictArray



class WorkerPredict(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,IdSharedMem,SM=None,Mode="Predict",iCluster=-1,DoSmearing=False):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.SM=SM
        self.IdSharedMem=IdSharedMem
        self.Mode=Mode
        self.iCluster=iCluster
        self.DoSmearing=DoSmearing

    def shutdown(self):
        self.exit.set()
    def run(self):
        while not self.kill_received:
            try:
                Row0,Row1 = self.work_queue.get()
            except:
                break

            D=NpShared.SharedToDico("%sDicoMemChunk"%self.IdSharedMem)
            DicoData={}
            DicoData["data"]=D["data"][Row0:Row1]
            DicoData["flags"]=D["flags"][Row0:Row1]
            DicoData["A0"]=D["A0"][Row0:Row1]
            DicoData["A1"]=D["A1"][Row0:Row1]
            DicoData["times"]=D["times"][Row0:Row1]
            DicoData["uvw"]=D["uvw"][Row0:Row1]
            DicoData["freqs"]=D["freqs"]
            DicoData["dfreqs"]=D["dfreqs"]
            # DicoData["UVW_dt"]=D["UVW_dt"]
            DicoData["infos"]=D["infos"]

            #DicoData["IndRows_All_UVW_dt"]=D["IndRows_All_UVW_dt"]
            #DicoData["All_UVW_dt"]=D["All_UVW_dt"]
            DicoData["UVW_dt"]=D["UVW_dt"][Row0:Row1]

            # DicoData["IndexTimesThisChunk"]=D["IndexTimesThisChunk"][Row0:Row1]
            # it0=np.min(DicoData["IndexTimesThisChunk"])
            # it1=np.max(DicoData["IndexTimesThisChunk"])+1
            # DicoData["UVW_RefAnt"]=D["UVW_RefAnt"][it0:it1,:,:]

            if "W" in D.keys():
                DicoData["W"]=D["W"][Row0:Row1]

            ApplyTimeJones=NpShared.SharedToDico("%sApplyTimeJones"%self.IdSharedMem)

            PM=ClassPredict(NCPU=1,DoSmearing=self.DoSmearing)

            #print DicoData.keys()


            if self.Mode=="Predict":
                PredictData=PM.predictKernelPolCluster(DicoData,self.SM,ApplyTimeJones=ApplyTimeJones)
                PredictArray=NpShared.GiveArray("%sPredictData"%(self.IdSharedMem))
                PredictArray[Row0:Row1]=PredictData[:]
            elif self.Mode=="ApplyCal":
                PM.ApplyCal(DicoData,ApplyTimeJones,self.iCluster)
            elif self.Mode=="GiveCovariance":
                PM.GiveCovariance(DicoData,ApplyTimeJones,self.SM)


            self.result_queue.put(True)




####################################################
####################################################



class ClassPredict():
    def __init__(self,Precision="S",NCPU=6,IdMemShared=None,DoSmearing="TF"):
        self.NCPU=NCPU
        ne.set_num_threads(self.NCPU)
        if Precision=="D":
            self.CType=np.complex128
            self.FType=np.float64
        if Precision=="S":
            self.CType=np.complex64
            self.FType=np.float32
        self.DoSmearing=DoSmearing

    
    # def ApplyCal(self,DicoData,ApplyTimeJones,iCluster):
    #     D=ApplyTimeJones
    #     Beam=D["Beam"]
    #     BeamH=D["BeamH"]
    #     lt0,lt1=D["t0"],D["t1"]
    #     ColOutDir=DicoData["data"]
    #     A0=DicoData["A0"]
    #     A1=DicoData["A1"]
    #     times=DicoData["times"]
    #     na=DicoData["infos"][0]

    #     # nt,nd,nd,nchan,_,_=Beam.shape
    #     # med=np.median(np.abs(Beam))
    #     # Threshold=med*1e-2
        
    #     for it in range(lt0.size):
            
    #         t0,t1=lt0[it],lt1[it]
    #         ind=np.where((times>=t0)&(times<t1))[0]

    #         if ind.size==0: continue
    #         data=ColOutDir[ind]
    #         # flags=DicoData["flags"][ind]
    #         A0sel=A0[ind]
    #         A1sel=A1[ind]
            
    #         if "ChanMap" in ApplyTimeJones.keys():
    #             ChanMap=ApplyTimeJones["ChanMap"]
    #         else:
    #             ChanMap=range(nf)

    #         for ichan in range(len(ChanMap)):
    #             JChan=ChanMap[ichan]
    #             if iCluster!=-1:
    #                 J0=Beam[it,iCluster,:,JChan,:,:].reshape((na,4))
    #                 JH0=BeamH[it,iCluster,:,JChan,:,:].reshape((na,4))
    #             else:
    #                 J0=np.mean(Beam[it,:,:,JChan,:,:],axis=1).reshape((na,4))
    #                 JH0=np.mean(BeamH[it,:,:,JChan,:,:],axis=1).reshape((na,4))
                
    #             J=ModLinAlg.BatchInverse(J0)
    #             JH=ModLinAlg.BatchInverse(JH0)

    #             data[:,ichan,:]=ModLinAlg.BatchDot(J[A0sel,:],data[:,ichan,:])
    #             data[:,ichan,:]=ModLinAlg.BatchDot(data[:,ichan,:],JH[A1sel,:])

    #             # Abs_g0=(np.abs(J0[A0sel,0])<Threshold)
    #             # Abs_g1=(np.abs(JH0[A1sel,0])<Threshold)
    #             # flags[Abs_g0,ichan,:]=True
    #             # flags[Abs_g1,ichan,:]=True

    #         ColOutDir[ind]=data[:]

    #         # DicoData["flags"][ind]=flags[:]


    def ApplyCal(self,DicoData,ApplyTimeJones,iCluster):
        D=ApplyTimeJones
        Beam=D["Beam"]
        BeamH=D["BeamH"]
        lt0,lt1=D["t0"],D["t1"]
        ColOutDir=DicoData["data"]
        A0=DicoData["A0"]
        A1=DicoData["A1"]
        times=DicoData["times"]
        na=DicoData["infos"][0]

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


    def GiveParamJonesList(self,DicoJonesMatrices,A0,A1):
        JonesMatrices=np.complex64(DicoJonesMatrices["Beam"])
        MapJones=np.int32(DicoJonesMatrices["MapJones"])
        #MapJones=np.int32(np.arange(A0.shape[0]))
        #print DicoJonesMatrices["MapJones"].shape
        #stop
        A0=np.int32(A0)
        A1=np.int32(A1)
        ParamJonesList=[MapJones,A0,A1,JonesMatrices]
        return ParamJonesList

    def GiveCovariance(self,DicoData,ApplyTimeJones,SM):
        D=ApplyTimeJones
        Beam=D["Beam"]
        BeamH=D["BeamH"]
        lt0,lt1=D["t0"],D["t1"]
        A0=DicoData["A0"]
        A1=DicoData["A1"]
        times=DicoData["times"]
        na=DicoData["infos"][0]


        #Predict=self.predictKernelPolCluster(DicoData,SM,ApplyTimeJones=ApplyTimeJones)

        Resid=DicoData["data"]#-Predict
        flags=DicoData["flags"]

        ListDirection=SM.Dirs

        #import pylab
        #pylab.clf()
        import ppgplot


        for iCluster in ListDirection:
            ParamJonesList=self.GiveParamJonesList(ApplyTimeJones,A0,A1)
            ParamJonesList=ParamJonesList+[iCluster]
            CVis=predict.CorrVis(Resid,ParamJonesList)

            ppgplot.pgopen('/xwin')
            ppgplot.pglab('(x)', '(y)', 'direction= %i\u2'%iCluster)
            ys=np.abs(CVis[flags==0])
            xs=np.arange(ys.size)
            ppgplot.pgenv(0.,np.max(xs),0.,np.max(ys),0,1)
            ppgplot.pgpt(xs,ys,1)
            ppgplot.pgclos()

            # pylab.plot(np.abs(CVis[flags==0]))
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
        return 

        MaxVis=predict.GiveMaxCorr(Resid,ParamJonesList)
        rms=findrms.findrms(MaxVis)
        med=np.median(MaxVis)

        W=DicoData["W"]
        diff=(MaxVis-med)/rms
        print rms
        diff[diff==0]=1e-6
        dev=1./(1.+diff)**2
        W[diff>3.]=0
        # W*=dev
        # W/=np.mean(W)
        


    # def GiveCovariance(self,DicoData,ApplyTimeJones):
    #     D=ApplyTimeJones
    #     Beam=D["Beam"]
    #     BeamH=D["BeamH"]
    #     lt0,lt1=D["t0"],D["t1"]
    #     A0=DicoData["A0"]
    #     A1=DicoData["A1"]
    #     times=DicoData["times"]
    #     na=DicoData["infos"][0]

    #     nt,nd,na,nch,_,_=Beam.shape
        
    #     W=DicoData["W"]#np.zeros((times.size,),np.float32)

    #     #print "tot",times.size
    #     for it in range(lt0.size):
    #         t0,t1=lt0[it],lt1[it]
    #         ind=np.where((times>=t0)&(times<t1))[0]
    #         if ind.size==0: continue
    #         #print it,t0,t1,ind.size
    #         #print "tot0",ind.size


    #         A0sel=A0[ind]
    #         A1sel=A1[ind]
            
    #         if "ChanMap" in ApplyTimeJones.keys():
    #             ChanMap=ApplyTimeJones["ChanMap"]
    #         else:
    #             ChanMap=range(nf)

    #         for ichan in range(len(ChanMap)):
    #             JChan=ChanMap[ichan]

    #             J=Beam[it,:,:,JChan,:,:].reshape((nd,na,4))
    #             JH=BeamH[it,:,:,JChan,:,:].reshape((nd,na,4))
                
    #             Jinv=ModLinAlg.BatchInverse(J)
    #             JHinv=ModLinAlg.BatchInverse(JH)
    #             W0=np.abs(ModLinAlg.BatchDot(Jinv[:,A0sel,:],JHinv[:,A1sel,:]))
    #             Wm=np.mean(np.abs(W0[:,:,0]),axis=0)**(-2)

    #             # gid=np.abs(J[:,A0sel,0])
    #             # gjd=np.abs(J[:,A1sel,0])
    #             # gi=np.mean(gid,axis=0)
    #             # gj=np.mean(gjd,axis=0)
    #             # p=0.01
    #             # Wm=p**2*(1./gi**2+1./gj**2+p**2/(gi*gj)**2)


    #             W[ind]=Wm[:]

    #     W/=np.mean(W)

    #     #W=W.reshape((W.size,1))*np.ones((1,4))
    #     #return W


    def predictKernelPolCluster(self,DicoData,SM,iDirection=None,ApplyJones=None,ApplyTimeJones=None,Noise=None):
        self.DicoData=DicoData
        self.SourceCat=SM.SourceCat
        self.SM=SM

        freq=DicoData["freqs"]
        times=DicoData["times"]
        nf=freq.size
        na=DicoData["infos"][0]
        
        nrows=DicoData["A0"].size
        DataOut=np.zeros((nrows,nf,4),self.CType)
        if nrows==0: return DataOut
        
        self.freqs=freq
        self.wave=299792458./self.freqs
        
        if iDirection!=None:
            ListDirection=[iDirection]
        else:
            ListDirection=SM.Dirs#range(SM.NDir)
        
        A0=DicoData["A0"]
        A1=DicoData["A1"]
        if ApplyJones!=None:
            na,NDir,_=ApplyJones.shape
            Jones=np.swapaxes(ApplyJones,0,1)
            Jones=Jones.reshape((NDir,na,4))
            JonesH=ModLinAlg.BatchH(Jones)

        TSmear=0.
        FSmear=0.

        if "T" in self.DoSmearing:
            TSmear=1.
        if "F" in self.DoSmearing:
            FSmear=1.
        # self.SourceCat.m[:]=0
        # self.SourceCat.l[:]=0.1
        # self.SourceCat.I[:]=10
        # self.SourceCat.alpha[:]=0

        # DataOut=DataOut[1:2]
        # self.DicoData["uvw"]=self.DicoData["uvw"][1:2]
        # self.DicoData["A0"]=self.DicoData["A0"][1:2]
        # self.DicoData["A1"]=self.DicoData["A1"][1:2]
        # self.DicoData["IndexTimesThisChunk"]=self.DicoData["IndexTimesThisChunk"][1:2]
        # self.SourceCat=self.SourceCat[0:1]

        DT=DicoData["infos"][1]
        UVW_dt=DicoData["UVW_dt"]
        
        ColOutDir=np.zeros(DataOut.shape,np.complex64)
        for iCluster in ListDirection:
            ColOutDir.fill(0)

            indSources=np.where(self.SourceCat.Cluster==iCluster)[0]
            T=ClassTimeIt("predict")
            T.disable()
            ### new
            SourceCat=self.SourceCat[indSources].copy()
            l=np.float32(SourceCat.l)
            m=np.float32(SourceCat.m)
            I=np.float32(SourceCat.I)
            alpha=np.float32(SourceCat.alpha)
            WaveL=np.float32(299792458./self.freqs)
            flux=np.float32(SourceCat.I)
            alpha=SourceCat.alpha
            dnu=np.float32(self.DicoData["dfreqs"])
            f0=(self.freqs/SourceCat.RefFreq[0])
            fluxFreq=np.float32(flux.reshape((flux.size,1))*(f0.reshape((1,f0.size)))**(alpha.reshape((alpha.size,1))))
            

            LSM=[l,m,fluxFreq]
            LFreqs=[WaveL,np.float32(self.freqs),dnu]
            LUVWSpeed=[UVW_dt,DT]

            LSmearMode=[FSmear,TSmear]
            T.timeit("init")

            if ApplyTimeJones!=None:
                ParamJonesList=self.GiveParamJonesList(ApplyTimeJones,A0,A1)
                ParamJonesList=ParamJonesList+[iCluster]
                predict.predictJones(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,ParamJonesList)
            else:
                predict.predict(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode)

            T.timeit("predict0")

            if Noise!=None:
                ColOutDir+=Noise*(np.random.randn(*ColOutDir.shape)+1j*np.random.randn(*ColOutDir.shape))
            DataOut+=ColOutDir


        return DataOut







    # def predictKernelPolCluster(self,DicoData,SM,iDirection=None,ApplyJones=None,ApplyTimeJones=None,Noise=None):
    #     self.DicoData=DicoData
    #     self.SourceCat=SM.SourceCat
    #     self.SM=SM

    #     freq=DicoData["freqs"]
    #     times=DicoData["times"]
    #     nf=freq.size
    #     na=DicoData["infos"][0]
        
    #     nrows=DicoData["A0"].size
    #     DataOut=np.zeros((nrows,nf,4),self.CType)
    #     if nrows==0: return DataOut
        
    #     self.freqs=freq
    #     self.wave=299792458./self.freqs
        
    #     if iDirection!=None:
    #         ListDirection=[iDirection]
    #     else:
    #         ListDirection=SM.Dirs#range(SM.NDir)
        
    #     A0=DicoData["A0"]
    #     A1=DicoData["A1"]
    #     if ApplyJones!=None:
    #         na,NDir,_=ApplyJones.shape
    #         Jones=np.swapaxes(ApplyJones,0,1)
    #         Jones=Jones.reshape((NDir,na,4))
    #         JonesH=ModLinAlg.BatchH(Jones)

    #     TSmear=0.
    #     FSmear=0.

    #     if "T" in self.DoSmearing:
    #         TSmear=1.
    #     if "F" in self.DoSmearing:
    #         FSmear=1.
    #     # self.SourceCat.m[:]=0
    #     # self.SourceCat.l[:]=0.1
    #     # self.SourceCat.I[:]=10
    #     # self.SourceCat.alpha[:]=0

    #     # DataOut=DataOut[1:2]
    #     # self.DicoData["uvw"]=self.DicoData["uvw"][1:2]
    #     # self.DicoData["A0"]=self.DicoData["A0"][1:2]
    #     # self.DicoData["A1"]=self.DicoData["A1"][1:2]
    #     # self.DicoData["IndexTimesThisChunk"]=self.DicoData["IndexTimesThisChunk"][1:2]
    #     # self.SourceCat=self.SourceCat[0:1]

    #     DT=DicoData["infos"][1]
    #     UVW_dt=DicoData["UVW_dt"]
        
    #     for iCluster in ListDirection:
    #         indSources=np.where(self.SourceCat.Cluster==iCluster)[0]
    #         ColOutDir=np.zeros(DataOut.shape,np.complex64)

    #         T=ClassTimeIt("predict")
    #         T.disable()
    #         ### new
    #         SourceCat=self.SourceCat[indSources].copy()
    #         l=np.float32(SourceCat.l)
    #         m=np.float32(SourceCat.m)
    #         I=np.float32(SourceCat.I)
    #         alpha=np.float32(SourceCat.alpha)
    #         WaveL=np.float32(299792458./self.freqs)
    #         flux=np.float32(SourceCat.I)
    #         alpha=SourceCat.alpha
    #         dnu=np.float32(self.DicoData["dfreqs"])
    #         f0=(self.freqs/SourceCat.RefFreq[0])
    #         fluxFreq=np.float32(flux.reshape((flux.size,1))*(f0.reshape((1,f0.size)))**(alpha.reshape((alpha.size,1))))
            

    #         LSM=[l,m,fluxFreq]
    #         LFreqs=[WaveL,np.float32(self.freqs),dnu]
    #         LUVWSpeed=[UVW_dt,DT]

    #         LSmearMode=[FSmear,TSmear]
    #         T.timeit("init")
            
    #         predict.predict(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode)
    #         T.timeit("predict0")
    #         ###### test
    #         # ColOutDir0=ColOutDir.copy()
    #         # ColOutDir.fill(0)

    #         # T.reinit()
    #         # for iSource in range(indSources.size):
    #         #     out=self.PredictDirSPW(iCluster,iSource)
    #         #     if type(out)==type(None): continue
    #         #     ColOutDir+=out
    #         # T.timeit("predict1")

    #         # #print ColOutDir0-ColOutDir
    #         # print np.max(ColOutDir0-ColOutDir)

    #         # stop
            
    #         # print iCluster,ListDirection
    #         # print ColOutDir.shape
    #         # ColOutDir.fill(0)
    #         # print ColOutDir.shape
    #         # ColOutDir[:,:,0]=1
    #         # print ColOutDir.shape
    #         # ColOutDir[:,:,3]=1
    #         # print ColOutDir.shape

    #         # Apply Jones
    #         if ApplyJones!=None:

    #             J=Jones[iCluster]
    #             JH=JonesH[iCluster]
    #             for ichan in range(nf):
    #                 ColOutDir[:,ichan,:]=ModLinAlg.BatchDot(J[A0,:],ColOutDir[:,ichan,:])
    #                 ColOutDir[:,ichan,:]=ModLinAlg.BatchDot(ColOutDir[:,ichan,:],JH[A1,:])

    #         if ApplyTimeJones!=None:#"DicoBeam" in DicoData.keys():
    #             D=ApplyTimeJones#DicoData["DicoBeam"]
    #             Beam=D["Beam"]
    #             BeamH=D["BeamH"]

    #             lt0,lt1=D["t0"],D["t1"]


    #             for it in range(lt0.size):
    #                 t0,t1=lt0[it],lt1[it]
    #                 ind=np.where((times>=t0)&(times<t1))[0]
    #                 if ind.size==0: continue
    #                 data=ColOutDir[ind]
                    
    #                 A0sel=A0[ind]
    #                 A1sel=A1[ind]

    #                 if "ChanMap" in ApplyTimeJones.keys():
    #                     ChanMap=ApplyTimeJones["ChanMap"]
    #                 else:
    #                     ChanMap=range(nf)


    #                 for ichan in range(len(ChanMap)):
    #                     JChan=ChanMap[ichan]

    #                     J=Beam[it,iCluster,:,JChan,:,:].reshape((na,4))
    #                     JH=BeamH[it,iCluster,:,JChan,:,:].reshape((na,4))
    #                     data[:,ichan,:]=ModLinAlg.BatchDot(J[A0sel,:],data[:,ichan,:])
    #                     data[:,ichan,:]=ModLinAlg.BatchDot(data[:,ichan,:],JH[A1sel,:])
    #                 ColOutDir[ind]=data[:]


    #         if Noise!=None:
    #             ColOutDir+=Noise*(np.random.randn(*ColOutDir.shape)+1j*np.random.randn(*ColOutDir.shape))
    #         DataOut+=ColOutDir


    #     return DataOut



