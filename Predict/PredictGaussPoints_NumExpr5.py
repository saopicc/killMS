import numpy as np
from pyrap.tables import table
from killMS2.Data.ClassMS import ClassMS
#from Sky.ClassSM import ClassSM
from killMS2.Other.ClassTimeIt import ClassTimeIt
import numexpr as ne
#import ModNumExpr
from killMS2.Other.progressbar import ProgressBar
import multiprocessing
from killMS2.Array import ModLinAlg
from killMS2.Array import NpShared
#ne.evaluate=lambda sin: ("return %s"%sin)
import time
from killMS2.Predict import predict 
from killMS2.Predict import predict_np19 
from killMS2.Other import findrms
from killMS2.Other.ModChanEquidistant import IsChanEquidistant
from DDFacet.Imager import ClassDDEGridMachine

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





####################################################
####################################################



class ClassPredict():
    def __init__(self,Precision="S",NCPU=6,IdMemShared=None,DoSmearing=""):
        self.NCPU=NCPU
        ne.set_num_threads(self.NCPU)
        if Precision=="D":
            self.CType=np.complex128
            self.FType=np.float64
        if Precision=="S":
            self.CType=np.complex64
            self.FType=np.float32
        self.DoSmearing=DoSmearing
        self.IdSharedMem=IdMemShared
    


    def ApplyCal(self,DicoData,ApplyTimeJones,iCluster):
        D=ApplyTimeJones
        Jones=D["Jones"]
        JonesH=D["JonesH"]
        lt0,lt1=D["t0"],D["t1"]
        ColOutDir=DicoData["data"]
        A0=DicoData["A0"]
        A1=DicoData["A1"]
        times=DicoData["times"]
        na=DicoData["infos"][0]

        
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
                    J0=Jones[it,iCluster,:,JChan,:,:].reshape((na,4))
                    JH0=JonesH[it,iCluster,:,JChan,:,:].reshape((na,4))
                else:
                    J0=np.mean(Jones[it,:,:,JChan,:,:],axis=1).reshape((na,4))
                    JH0=np.mean(JonesH[it,:,:,JChan,:,:],axis=1).reshape((na,4))
                
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


    def GiveParamJonesList(self,DicoJonesMatricesIn,A0,A1):
        if "DicoApplyJones" in DicoJonesMatricesIn.keys():
            DicoJonesMatrices=DicoJonesMatricesIn["DicoApplyJones"]
        else:
            DicoJonesMatrices=DicoJonesMatricesIn
        JonesMatrices=np.complex64(DicoJonesMatrices["Jones"])
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


        #print "predict...",times[0],times[-1]
        #Predict=self.predictKernelPolCluster(DicoData,SM,ApplyTimeJones=ApplyTimeJones)
        #print "done..."


        Resid=DicoData["resid"]#-Predict
        flags=DicoData["flags"]

        ListDirection=SM.Dirs

        #import pylab
        #pylab.clf()
        import ppgplot

        MaxMat=np.zeros(Resid.shape,dtype=np.float32)
        CVis=np.zeros_like(Resid)

        for iCluster in ListDirection:
            CVis.fill(0)
            ParamJonesList=self.GiveParamJonesList(ApplyTimeJones,A0,A1)
            ParamJonesList=ParamJonesList+[iCluster]
            predict.CorrVis(Resid,CVis,ParamJonesList)

            CVis[flags==1]=0.
            aCVis=np.abs(CVis)
            
            #print "In direction %i: (std, std_abs, med_abs, max_abs)=(%f, %f, %f, %f)"%(iCluster,np.std(CVis),np.std(aCVis),np.median(aCVis),np.max(aCVis))
            ind=(aCVis>MaxMat)
            MaxMat[ind]=aCVis[ind]

            del(aCVis)


        nrow,nch,_=Resid.shape
        W=DicoData["W"]#np.ones((nrow,nch),np.float32)
        rms=findrms.findrms(MaxMat)
        diff=np.abs(MaxMat-np.median(MaxMat))/rms
        cond=(diff>3.)
        ind=np.any(cond,axis=2)
        W[ind]=0.



    def GiveGM(self,iFacet,SM):
        GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(SM.GD,
                                                            SM.DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                                                            SM.DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                                                            lmShift=SM.DicoImager[iFacet]["lmShift"],
                                                            IdSharedMem=self.IdSharedMem,IDFacet=SM.DicoImager[iFacet]["IDFacet"],
                                                            SpheNorm=False)
        return GridMachine

    def InitGM(self,SM):
        self.DicoGM={}
        LFacets=SM.DicoImager.keys()
        for iFacet in LFacets:
            self.DicoGM[iFacet]=self.GiveGM(iFacet,SM)


    def predictKernelPolCluster(self,DicoData,SM,**kwargs):
        if SM.Type=="Catalog":
            return self.predictKernelPolClusterCatalog(DicoData,SM,**kwargs)
        elif SM.Type=="Image":
            return self.predictKernelPolClusterImage(DicoData,SM,**kwargs)



    def predictKernelPolClusterCatalog(self,DicoData,SM,iDirection=None,ApplyJones=None,ApplyTimeJones=None,Noise=None):
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


        if self.DoSmearing!=0:
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

            AllowEqualiseChan=IsChanEquidistant(DicoData["freqs"])

            if ApplyTimeJones!=None:

                ParamJonesList=self.GiveParamJonesList(ApplyTimeJones,A0,A1)
                ParamJonesList=ParamJonesList+[iCluster]
                #predict.predictJones(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,ParamJonesList)

                #ColOutDir.fill(0)
                #predict.predictJones(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,ParamJonesList,0)
                #d0=ColOutDir.copy()
                #ColOutDir.fill(0)

                # predict.predictJones(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,ParamJonesList,AllowEqualiseChan)
                # ColOutDir0=ColOutDir.copy()
                # ColOutDir.fill(0)

                #predict.predictJones2(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,ParamJonesList,AllowEqualiseChan)
                print LSmearMode
                predict.predictJones2(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,AllowEqualiseChan)
                predict.ApplyJones(ColOutDir,ParamJonesList)

                # print ColOutDir

                #d1=ColOutDir.copy()
                #ind=np.where(d0!=0)
                #print np.max((d0-d1)[ind]/(d0[ind]))
                #stop
            else:
                #predict.predict(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode)
                #AllowEqualiseChan=0
                #predict.predict(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,AllowEqualiseChan)
                #d0=ColOutDir.copy()
                #ColOutDir.fill(0)

                
                predict.predict(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,AllowEqualiseChan)

                # d0=ColOutDir.copy()
                # ColOutDir.fill(0)

                # predict_np19.predict(ColOutDir,(DicoData["uvw"]),LFreqs,LSM,LUVWSpeed,LSmearMode,AllowEqualiseChan)
                # print ColOutDir,d0
                # d1=ColOutDir.copy()
                # ind=np.where(d0!=0)
                # print np.max((d0-d1)[ind]/(d0[ind]))
                # stop


            del(l,m,I,SourceCat,alpha,WaveL,flux,dnu,f0,fluxFreq,LSM,LFreqs)
                


                # d1=ColOutDir
                # ind=np.where(d0!=0)
                # print np.max((d0-d1)[ind]/(d0[ind]))
                # stop
                

            T.timeit("predict0")

            if Noise!=None:
                ColOutDir+=Noise*(np.random.randn(*ColOutDir.shape)+1j*np.random.randn(*ColOutDir.shape))
                stop
            DataOut+=ColOutDir

        #del(LFreqs,LSM,LUVWSpeed,LSmearMode)
        #del(ColOutDir)

        return DataOut


    ######################################################
    ######################################################
    ######################################################


    def predictKernelPolClusterImage(self,DicoData,SM,iDirection=None,ApplyJones=None,ApplyTimeJones=None,Noise=None):
        T=ClassTimeIt("predictKernelPolClusterImage")
        T.disable()
        self.DicoData=DicoData
        self.SM=SM

        
        freq=DicoData["freqs"]
        times=DicoData["times"]
        nf=freq.size
        na=DicoData["infos"][0]
        
        nrows=DicoData["A0"].size
        T.timeit("0")
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
        # if ApplyJones!=None:
        #     na,NDir,_=ApplyJones.shape
        #     Jones=np.swapaxes(ApplyJones,0,1)
        #     Jones=Jones.reshape((NDir,na,4))
        #     JonesH=ModLinAlg.BatchH(Jones)

        TSmear=0.
        FSmear=0.


        if self.DoSmearing!=0:
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
        DATA=DicoData

        T.timeit("1")


        for iCluster in ListDirection:
            iFacet=iCluster
            if SM.ClusterCat.SumI[iFacet]==0: continue
            GridMachine=self.DicoGM[iFacet]#self.GiveGM(iFacet,SM)
            T.timeit("2: GM")
            uvwThis=DATA["uvw"]
            flagsThis=DATA["flags_image"]
            times=DATA["times"]
            A0=DATA["A0"]
            A1=DATA["A1"]
            A0A1=A0,A1
            freqs=DATA["freqs"]

            # DicoJonesMatrices=None
            DicoJonesMatrices=ApplyTimeJones

            ModelIm = NpShared.UnPackListArray("%sGrids"%self.IdSharedMem)[iFacet]
            T.timeit("2: Stuff")
            vis=GridMachine.get(times,uvwThis,ColOutDir,flagsThis,A0A1,ModelIm,DicoJonesMatrices=DicoJonesMatrices,freqs=freqs,
                                ImToGrid=False)
            T.timeit("2: Predict")
            # get() is substracting
            DataOut-=ColOutDir
            ColOutDir.fill(0)
            T.timeit("2: End")


        return DataOut


#####################################################


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

            if "resid" in D.keys():
                DicoData["resid"]=D["resid"][Row0:Row1]

            ApplyTimeJones=NpShared.SharedToDico("%sApplyTimeJones"%self.IdSharedMem)
            #JonesMatrices=ApplyTimeJones["Beam"]
            #print ApplyTimeJones["Beam"].flags
            ApplyTimeJones["MapJones"]=ApplyTimeJones["MapJones"][Row0:Row1]
            
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
