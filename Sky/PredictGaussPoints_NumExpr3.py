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

    def GiveCovariance(self,DicoDataIn,ApplyTimeJones):

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
            W=WorkerPredict(work_queue, result_queue,self.IdMemShared,Mode="GiveCovariance",DoSmearing=self.DoSmearing)
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
            DicoData["infos"]=D["infos"]
            DicoData["IndexTimesThisChunk"]=D["IndexTimesThisChunk"][Row0:Row1]

            it0=np.min(DicoData["IndexTimesThisChunk"])
            it1=np.max(DicoData["IndexTimesThisChunk"])+1
            DicoData["UVW_RefAnt"]=D["UVW_RefAnt"][it0:it1,:,:]

            if "W" in D.keys():
                DicoData["W"]=D["W"][Row0:Row1]

            ApplyTimeJones=NpShared.SharedToDico("%sApplyTimeJones"%self.IdSharedMem)

            PM=ClassPredict(NCPU=1,DoSmearing=self.DoSmearing)

            if self.Mode=="Predict":
                PredictData=PM.predictKernelPolCluster(DicoData,self.SM,ApplyTimeJones=ApplyTimeJones)
                PredictArray=NpShared.GiveArray("%sPredictData"%(self.IdSharedMem))
                PredictArray[Row0:Row1]=PredictData[:]
            elif self.Mode=="ApplyCal":
                PM.ApplyCal(DicoData,ApplyTimeJones,self.iCluster)
            elif self.Mode=="GiveCovariance":
                PM.GiveCovariance(DicoData,ApplyTimeJones)


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


    def GiveCovariance(self,DicoData,ApplyTimeJones):
        D=ApplyTimeJones
        Beam=D["Beam"]
        BeamH=D["BeamH"]
        lt0,lt1=D["t0"],D["t1"]
        A0=DicoData["A0"]
        A1=DicoData["A1"]
        times=DicoData["times"]
        na=DicoData["infos"][0]

        nt,nd,na,nch,_,_=Beam.shape
        
        W=DicoData["W"]#np.zeros((times.size,),np.float32)
        #print "tot",times.size
        for it in range(lt0.size):
            t0,t1=lt0[it],lt1[it]
            ind=np.where((times>=t0)&(times<t1))[0]
            if ind.size==0: continue
            #print it,t0,t1,ind.size
            #print "tot0",ind.size


            A0sel=A0[ind]
            A1sel=A1[ind]
            
            if "ChanMap" in ApplyTimeJones.keys():
                ChanMap=ApplyTimeJones["ChanMap"]
            else:
                ChanMap=range(nf)

            for ichan in range(len(ChanMap)):
                JChan=ChanMap[ichan]

                J=Beam[it,:,:,JChan,:,:].reshape((nd,na,4))
                JH=BeamH[it,:,:,JChan,:,:].reshape((nd,na,4))
                
                Jinv=ModLinAlg.BatchInverse(J)
                JHinv=ModLinAlg.BatchInverse(JH)
                W0=np.abs(ModLinAlg.BatchDot(Jinv[:,A0sel,:],JHinv[:,A1sel,:]))
                Wm=np.mean(np.abs(W0[:,:,0]),axis=0)**(-2)

                # gid=np.abs(J[:,A0sel,0])
                # gjd=np.abs(J[:,A1sel,0])
                # gi=np.mean(gid,axis=0)
                # gj=np.mean(gjd,axis=0)
                # p=0.01
                # Wm=p**2*(1./gi**2+1./gj**2+p**2/(gi*gj)**2)


                W[ind]=Wm[:]

        W/=np.mean(W)

        #W=W.reshape((W.size,1))*np.ones((1,4))
        #return W







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

        for iCluster in ListDirection:
            indSources=np.where(self.SourceCat.Cluster==iCluster)[0]
            ColOutDir=np.zeros_like(DataOut)
            for iSource in range(indSources.size):
                out=self.PredictDirSPW(iCluster,iSource)
                if type(out)==type(None): continue
                ColOutDir+=out
            
            # print iCluster,ListDirection
            # print ColOutDir.shape
            # ColOutDir.fill(0)
            # print ColOutDir.shape
            # ColOutDir[:,:,0]=1
            # print ColOutDir.shape
            # ColOutDir[:,:,3]=1
            # print ColOutDir.shape

            # Apply Jones
            if ApplyJones!=None:

                J=Jones[iCluster]
                JH=JonesH[iCluster]
                for ichan in range(nf):
                    ColOutDir[:,ichan,:]=ModLinAlg.BatchDot(J[A0,:],ColOutDir[:,ichan,:])
                    ColOutDir[:,ichan,:]=ModLinAlg.BatchDot(ColOutDir[:,ichan,:],JH[A1,:])

            if ApplyTimeJones!=None:#"DicoBeam" in DicoData.keys():
                D=ApplyTimeJones#DicoData["DicoBeam"]
                Beam=D["Beam"]
                BeamH=D["BeamH"]

                lt0,lt1=D["t0"],D["t1"]


                for it in range(lt0.size):
                    t0,t1=lt0[it],lt1[it]
                    ind=np.where((times>=t0)&(times<t1))[0]
                    if ind.size==0: continue
                    data=ColOutDir[ind]
                    A0sel=A0[ind]
                    A1sel=A1[ind]

                    if "ChanMap" in ApplyTimeJones.keys():
                        ChanMap=ApplyTimeJones["ChanMap"]
                    else:
                        ChanMap=range(nf)

                    for ichan in range(len(ChanMap)):
                        JChan=ChanMap[ichan]

                        J=Beam[it,iCluster,:,JChan,:,:].reshape((na,4))
                        JH=BeamH[it,iCluster,:,JChan,:,:].reshape((na,4))
                        data[:,ichan,:]=ModLinAlg.BatchDot(J[A0sel,:],data[:,ichan,:])
                        data[:,ichan,:]=ModLinAlg.BatchDot(data[:,ichan,:],JH[A1sel,:])
                    ColOutDir[ind]=data[:]
            

            if Noise!=None:
                ColOutDir+=Noise*(np.random.randn(*ColOutDir.shape)+1j*np.random.randn(*ColOutDir.shape))
            DataOut+=ColOutDir


        return DataOut

    def GiveKp(self,DicoData,SM,idir=None,isource=None):
        SourceCat=SM.SourceCat
        
        if isource!=None:
            ind0=np.where(SourceCat.Cluster==idir)[0][isource:isource+1]
            SourceCat=SourceCat[ind0]
        elif idir!=None:
            ind0=np.where(SourceCat.Cluster==idir)[0]
            SourceCat=SourceCat[ind0]

        NSource=SourceCat.shape[0]
        if NSource==0: return None

        T=ClassTimeIt("GiveKp")
        T.disable()
        IndexTimesThisChunk=DicoData["IndexTimesThisChunk"]
        IndexTimesThisChunk_0=IndexTimesThisChunk-IndexTimesThisChunk[0]
        UVW_RefAnt=DicoData["UVW_RefAnt"]#[IndexTimesThisChunk_0]


        freqs=DicoData["freqs"]
        pi=np.pi
        wave=299792458./freqs

        # uvw=DicoData["uvw"]
        # U=self.FType(uvw[:,0].flatten().copy())
        # V=self.FType(uvw[:,1].flatten().copy())
        # W=self.FType(uvw[:,2].flatten().copy())
        # U=U.reshape((1,U.size,1,1))
        # V=V.reshape((1,U.size,1,1))
        # W=W.reshape((1,U.size,1,1))

        f0=self.CType(2*pi*1j/wave)
        f0=f0.reshape((1,1,1,f0.size))
        nf=wave.size
        
        T.timeit("0")
        Ll=self.FType(SourceCat.l)
        Lm=self.FType(SourceCat.m)
        l=Ll.reshape(NSource,1,1,1)
        m=Lm.reshape(NSource,1,1,1)
        nn=self.FType(np.sqrt(1.-l**2-m**2)-1.)

        #KernelPha=ne.evaluate("f0*(U*l+V*m+W*nn)").astype(self.CType)

        # lm : [nd,nt,na,nf]
        nt,na,_=UVW_RefAnt.shape
        U_shape=(1,nt,na,1)
        U_refAnt=UVW_RefAnt[:,:,0].reshape(U_shape)# [1, nt, na, 1]
        V_refAnt=UVW_RefAnt[:,:,1].reshape(U_shape)
        W_refAnt=UVW_RefAnt[:,:,2].reshape(U_shape)

        T.timeit("1")
        # KpB=np.exp(-f0*(U*l+V*m+W*nn))
        # T.timeit("Kp0")
        # Kp0=np.exp(-f0*(U_refAnt*l+V_refAnt*m+W_refAnt*nn))
        # T.timeit("Kp1")
        nd=NSource

        CondEquidistant=False
        if freqs.size>2:
            dfs=freqs[1::]-freqs[0:-1]
            dfs-=dfs[0]
            dfs/=1e-6
            CondEquidistant=(np.max(np.abs(dfs))<1e-3)

        if (wave.size>2)&(CondEquidistant):
            Kp=np.zeros((nd,nt,na,nf),self.CType)
            Kp[:,:,:,0:1]=np.exp(-f0[:,:,:,0:1]*(U_refAnt*l+V_refAnt*m+W_refAnt*nn))
            df0=-f0[:,:,:,1]-(-f0[:,:,:,0])
            df0=df0.reshape((1,1,1,1))
            Kp_phase=(U_refAnt*l+V_refAnt*m+W_refAnt*nn)
            dKp=np.exp(df0[:,:,:,:]*Kp_phase)
            for ich in range(1,nf):
                Kp[:,:,:,ich]=Kp[:,:,:,ich-1]*dKp[:,:,:,0]
            T.timeit("Kp2")
        else:
            Kp_phase=(U_refAnt*l+V_refAnt*m+W_refAnt*nn)
            Kp=np.exp(-f0*Kp_phase)

        Kp_phase=(U_refAnt*l+V_refAnt*m+W_refAnt*nn)
            

        return Kp,Kp_phase


    def PredictDirSPW(self,idir,isource=None):

        IDSource=None
        if isource!=None:
            IDs=np.arange(self.SourceCat.shape[0])
            ind0=np.where(self.SourceCat.Cluster==idir)[0]
            IDs=IDs[ind0][isource:isource+1]
            ind0=ind0[isource:isource+1]
        else:
            ind0=np.where(self.SourceCat.Cluster==idir)[0]

        T=ClassTimeIt("PredictDirSPW",f=1e3)
        
        #print idir,isource
        T.disable()
        IndexTimesThisChunk=self.DicoData["IndexTimesThisChunk"]
        IndexTimesThisChunk_0=IndexTimesThisChunk-IndexTimesThisChunk[0]
        UVW_RefAnt=self.DicoData["UVW_RefAnt"]#[IndexTimesThisChunk_0]

        NSource=ind0.size
        if NSource==0: return None
        SourceCat=self.SourceCat[ind0]
        freq=self.freqs
        pi=np.pi
        wave=self.wave#[0]
        uvw=self.DicoData["uvw"]

                


        U=self.FType(uvw[:,0].flatten().copy())
        V=self.FType(uvw[:,1].flatten().copy())
        W=self.FType(uvw[:,2].flatten().copy())
        U=U.reshape((1,U.size,1,1))
        V=V.reshape((1,U.size,1,1))
        W=W.reshape((1,U.size,1,1))
    
        
        #ColOut=np.zeros(U.shape,dtype=complex)
        f0=self.CType(2*pi*1j/wave)
        f0=f0.reshape((1,1,1,f0.size))
        nf=wave.size
        rasel =SourceCat.ra
        decsel=SourceCat.dec
        
        T.timeit("0")
        Ll=self.FType(SourceCat.l)
        Lm=self.FType(SourceCat.m)
        l=Ll.reshape(NSource,1,1,1)
        m=Lm.reshape(NSource,1,1,1)
        nn=self.FType(np.sqrt(1.-l**2-m**2)-1.)

        #KernelPha=ne.evaluate("f0*(U*l+V*m+W*nn)").astype(self.CType)
        T.timeit("1")

        if "Kp" in self.DicoData.keys():
            nd,nt,na,nf=self.DicoData["Kp"].shape
            if NSource==1:
                Kp=self.DicoData["Kp"][IDs[0]].reshape(1,nt,na,nf)
            else:
                Kp=self.DicoData["Kp"][IDs]
        else:
            Kp,Kp_phase=self.GiveKp(self.DicoData,self.SM,idir=idir,isource=isource)

        # # # lm : [nd,nt,na,nf]
        # nt,na,_=UVW_RefAnt.shape
        # U_shape=(1,nt,na,1)
        # U_refAnt=UVW_RefAnt[:,:,0].reshape(U_shape)# [1, nt, na, 1]
        # V_refAnt=UVW_RefAnt[:,:,1].reshape(U_shape)
        # W_refAnt=UVW_RefAnt[:,:,2].reshape(U_shape)

        # # T.timeit("1")
        # # KpB=np.exp(-f0*(U*l+V*m+W*nn))
        # # T.timeit("Kp0")
        # Kp0=np.exp(-f0*(U_refAnt*l+V_refAnt*m+W_refAnt*nn))
        # # T.timeit("Kp1")
        # # nd=NSource

        # # dfs=self.freqs[1::]-self.freqs[0:-1]
        # # dfs-=dfs[0]
        # # dfs/=1e-6
        # # CondEquidistant=(np.max(np.abs(dfs))<1e-3)

        # # if (wave.size>2)&(CondEquidistant):
        # #     Kp=np.zeros((nd,nt,na,nf),self.CType)
        # #     Kp[:,:,:,0:1]=np.exp(-f0[:,:,:,0:1]*(U_refAnt*l+V_refAnt*m+W_refAnt*nn))
        # #     df0=-f0[:,:,:,1]-(-f0[:,:,:,0])
        # #     df0=df0.reshape((1,1,1,1))
        # #     dKp=np.exp(df0[:,:,:,:]*(U_refAnt*l+V_refAnt*m+W_refAnt*nn))
        # #     for ich in range(1,nf):
        # #         Kp[:,:,:,ich]=Kp[:,:,:,ich-1]*dKp[:,:,:,0]
        # #     T.timeit("Kp2")
        # # else:
        # #     Kp0=np.exp(-f0*(U_refAnt*l+V_refAnt*m+W_refAnt*nn))
        # print Kp0-Kp

        # stop

        T.timeit("2")
        A0=self.DicoData["A0"]
        A1=self.DicoData["A1"]

        indxTime=IndexTimesThisChunk_0
        KpRow=Kp[:,indxTime,A0,:]
        KqRow=Kp[:,indxTime,A1,:]
        # KpRow=Kp[:,indxTime[0]:indxTime[-1]+1,A0,:]
        # KqRow=Kp[:,indxTime,A1,:]
        # stop


        T.timeit("2a")
        Kpq=KpRow*KqRow.conj()
        T.timeit("2b")
        nrow=IndexTimesThisChunk_0.size
        Kpq=Kpq.reshape((NSource,nrow,nf,1))

        T.timeit("3")
        ##########################

        TypeSources=SourceCat.Type
        Gmaj=SourceCat.Gmaj.reshape((NSource,1,1,1))
        Gmin=SourceCat.Gmin.reshape((NSource,1,1,1))
        Gangle=SourceCat.Gangle.reshape((NSource,1,1,1))
        RefFreq=SourceCat.RefFreq.reshape((NSource,1,1,1))
        alpha=SourceCat.alpha.reshape((NSource,1,1,1))
        fI=SourceCat.I.reshape((NSource,1,1))
        fQ=SourceCat.Q.reshape((NSource,1,1))
        fU=SourceCat.U.reshape((NSource,1,1))
        fV=SourceCat.V.reshape((NSource,1,1))
        Sky=np.zeros((NSource,1,1,4),self.CType)
        Sky[:,:,:,0]=(fI+fQ);
        Sky[:,:,:,1]=(fU+1j*fV);
        Sky[:,:,:,2]=(fU-1j*fV);
        Sky[:,:,:,3]=(fI-fQ);

        T.timeit("4")
        Ssel  =Sky*(freq.reshape((1,1,freq.size,1))/RefFreq)**(alpha)
        Ssel=self.CType(Ssel)
        f=Ssel
        Ssel[Ssel==0]=1e-10
        T.timeit("4a")

        ################

        Kpq=Kpq*Ssel
        # T.timeit("5")
        # Kpq0=np.zeros_like(Kpq)
        # for ipol in range(4):
        #     Kpq0[:,:,:,ipol]=Kpq[:,:,:,0]*Ssel[:,:,:,ipol]
        # T.timeit("5a")

        #################
        indGauss=np.where(TypeSources==1)[0]
        NGauss=indGauss.size
        
        if NGauss>0:
            ang=Gangle[indGauss].reshape((NGauss,1,1,1))
            SigMaj=Gmaj[indGauss].reshape((NGauss,1,1,1))
            SigMin=Gmin[indGauss].reshape((NGauss,1,1,1))
            WaveL=wave
            SminCos=SigMin*np.cos(ang)
            SminSin=SigMin*np.sin(ang)
            SmajCos=SigMaj*np.cos(ang)
            SmajSin=SigMaj*np.sin(ang)
            #up=ne.evaluate("U*SminCos-V*SminSin")
            #vp=ne.evaluate("U*SmajSin+V*SmajCos")
            const=-(2*(pi**2)*(1/WaveL)**2)#*fudge
            const=const.reshape((1,1,freq.size,1))
            uvp=ne.evaluate("exp(const*((U*SminCos-V*SminSin)**2+(U*SmajSin+V*SmajCos)**2))")
            #KernelPha=ne.evaluate("KernelPha+uvp")
            Kpq[indGauss,:,:,:]*=uvp[:,:,:,:]
        T.timeit("6")
        
        if self.DoSmearing!=None:
            if "F" in self.DoSmearing:
                dfreqs=self.DicoData["dfreqs"]
                KpRow_Phase=Kp_phase[:,indxTime,A0,:]
                KqRow_Phase=Kp_phase[:,indxTime,A1,:]
                dfreqs=dfreqs.copy().reshape((1,1,1,dfreqs.size))/299792458.
                dphi=(2.*np.pi)*(KpRow_Phase-KqRow_Phase)*dfreqs # (nd=1,nt,na,nf=1)
                decorr=np.sinc(dphi/2.).reshape((NSource,nrow,nf,1))
                Kpq=Kpq*decorr
            # if "T" in self.DoSmearing:
            #     dfreqs=self.DicoData["dfreqs"]
            #     KpRow_Phase=Kp_phase[:,indxTime,A0,:]
            #     KqRow_Phase=Kp_phase[:,indxTime,A1,:]
            #     dfreqs=dfreqs.copy().reshape((1,1,1,dfreqs.size))/299792458.
            #     dphi=(2.*np.pi)*(KpRow_Phase-KqRow_Phase)*dfreqs # (nd=1,nt,na,nf=1)
            #     decorr=np.sinc(dphi/2.).reshape((NSource,nrow,nf,1))
            #     Kpq=Kpq*decorr

        

        #Kernel=ne.evaluate("f*exp(KernelPha)").astype(self.CType)

        if Kpq.shape[0]>1:
            ColOut=ne.evaluate("sum(Kpq,axis=0)").astype(self.CType)
        else:
            ColOut=Kpq[0]
        T.timeit("6")

        return ColOut
