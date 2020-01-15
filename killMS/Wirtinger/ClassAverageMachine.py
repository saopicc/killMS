from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

class ClassAverageMachine():
    def __init__(self,
                 GD,
                 PM_Compress,
                 SM_Compress,
                 PolMode="Scalar",
                 DicoMergeStations=None):
        self.GD=GD
        self.PM_Compress=PM_Compress
        self.SM_Compress=SM_Compress
        self.PolMode=PolMode
        self.DicoMergeStations=DicoMergeStations
        if self.GD["Solvers"]["NChanSols"]!=1:
            raise ValueError("Only deal with NChanSol=1")
        if self.GD["Solvers"]["PolMode"]!="Scalar":
            raise ValueError("Only deal with PolMode=Scalar")
        self.NJacobBlocks_X,self.NJacobBlocks_Y=1,1
        
    def AverageKernelMatrix(self,DicoData,K):
        #A0,A1=DicoData[""]
        A0=DicoData["A0"]
        A1=DicoData["A1"]
            
        NDir,Np,Npol=K.shape

        NpBlBlocks=DicoData["NpBlBlocks"][0]
        A0A1=sorted(list(set([(A0[i],A1[i]) for i in range(A0.size)])))
        
        
        NpOut=len(A0A1)
        NDirAvg=self.SM_Compress.NDir
        KOut=np.zeros((NDir,NDirAvg,NpOut,Npol),K.dtype)
        
        IndList=[(np.where((A0==ThisA0)&(A1==ThisA1))[0]) for (ThisA0,ThisA1) in A0A1]
        
        for iDirAvg in range(NDirAvg):
            K_Compress=self.PM_Compress.predictKernelPolCluster(DicoData,self.SM_Compress,iDirection=iDirAvg)
            for iDir in range(NDir):
                p=K[iDir,:,:]
                pp=p*K_Compress[:,:,0].conj()
                for iBl,ind in enumerate(IndList):
                    KOut[iDir,iDirAvg,iBl,0]=np.mean(pp[ind])
        
        KOut=KOut.reshape((NDir,NDirAvg*NpOut,Npol))
        KOut[:,:,3]=KOut[:,:,0]
        
        return KOut

    
    
    def MergeAntennaJacobian(self,DicoData,LJacob):
        A0=DicoData["A0_Avg"]
        A1=DicoData["A1_Avg"]

        LJacobOut=[]
        indBlNonMerge=DicoData["indBlNonMerge"]
        indBlMerge=DicoData["indBlMerge"]
        if indBlMerge.size==0: return LJacob
        for J in LJacob:
            n4vis,NDir=J.shape
            JOut=np.zeros((indBlNonMerge.size+1,NDir),dtype=J.dtype)
            j=J[indBlMerge,:]
            w=1.-DicoData["flags_flat_avg"].flat[indBlMerge].reshape((-1,1))
            sw=np.sum(w)
            JOut[0,:]=np.sum(j*w,axis=0)/sw
            JOut[1:,:]=J[indBlNonMerge,:]
            LJacobOut.append(JOut)
        return LJacobOut
    
    def AverageDataVector(self,DicoData):
        A0=DicoData["A0"].ravel()
        A1=DicoData["A1"].ravel()
        

        NpBlBlocks=DicoData["NpBlBlocks"][0]
        A0A1=sorted(list(set([(A0[i],A1[i]) for i in range(A0.size)])))
        NpOut=len(A0A1)
        NDirAvg=self.SM_Compress.NDir

        IndList=[(np.where((A0==ThisA0)&(A1==ThisA1))[0]) for (ThisA0,ThisA1) in A0A1]

        DicoData["A0_Avg"]=np.array([A0A1[i][0] for i in range(len(A0A1))]*NDirAvg)
        DicoData["A1_Avg"]=np.array([A0A1[i][1] for i in range(len(A0A1))]*NDirAvg)
        
        
        d=DicoData["data"]
        f=DicoData["flags"]
        nr,nch,_,_=d.shape
        
        DOut = np.zeros((NDirAvg,NpOut,1,1),d.dtype)
        FOut=np.zeros(DOut.shape,f.dtype)

        for iDirAvg in range(NDirAvg):
            K_Compress=self.PM_Compress.predictKernelPolCluster(DicoData,self.SM_Compress,iDirection=iDirAvg)
            dp=d[:,:,0,0]*K_Compress[:,:,0].conj()
            
            for iBl,ind in enumerate(IndList):
                if np.min(f[ind])==1:
                    FOut[iDirAvg,iBl,0]=1
                    continue
                w=(1-f[ind])
                DOut[iDirAvg,iBl,0]=np.sum(dp[ind].ravel()*w.ravel())/np.float32(np.sum(w))
        
        DicoData["flags_avg"]=FOut
        DicoData["data_avg"]=DOut
        
        DicoData["flags_flat_avg"]=np.rollaxis(FOut,2).reshape(self.NJacobBlocks_X,NDirAvg*NpOut*self.NJacobBlocks_Y)
        DicoData["data_flat_avg"]=np.rollaxis(DOut,2).reshape(self.NJacobBlocks_X,NDirAvg*NpOut*self.NJacobBlocks_Y)

        if self.DicoMergeStations:
            indBlMerge=[]
            indBlNonMerge=[]
            A0=DicoData["A0_Avg"]
            A1=DicoData["A1_Avg"]
            n4vis=A0.size
            for iBl in range(n4vis):
                a0a1=(A0[iBl],A1[iBl])
                if not a0a1 in self.DicoMergeStations["ListBLMerge"]:
                    indBlNonMerge.append(iBl)
                else:
                    indBlMerge.append(iBl)


            indBlMerge=np.array(indBlMerge)
            indBlNonMerge=np.array(indBlNonMerge)
            
            DicoData["indBlMerge"]=indBlMerge
            DicoData["indBlNonMerge"]=indBlNonMerge

            flags_flat_avg=DicoData["flags_flat_avg"]
            data_flat_avg=DicoData["data_flat_avg"]

            if indBlMerge.size!=0:
                w=1.-flags_flat_avg.flat[indBlMerge]
                d=data_flat_avg.flat[indBlMerge]
                sw=np.sum(w)
                #print w,sw
                
                data_flat_avg_merged=np.zeros((1,indBlNonMerge.size+1),DicoData["data_flat_avg"].dtype)
                data_flat_avg_merged.flat[0]=np.sum(d*w)/sw
                data_flat_avg_merged.flat[1:]=data_flat_avg.flat[indBlNonMerge]
                
                flags_flat_avg_merged=np.zeros((1,indBlNonMerge.size+1),DicoData["flags_flat_avg"].dtype)
                flags_flat_avg_merged.flat[0]=0
                flags_flat_avg_merged.flat[1:]=flags_flat_avg.flat[indBlNonMerge]


                # flags_flat_merged=DicoData["flags_avg"][:,]
            else:
                flags_flat_avg_merged=flags_flat_avg
                data_flat_avg_merged=data_flat_avg
                
        else:
            flags_flat_avg_merged=DicoData["flags_flat_avg"]
            data_flat_avg_merged=DicoData["data_flat_avg"]

        DicoData["flags_flat_avg_merged"]=flags_flat_avg_merged
        DicoData["data_flat_avg_merged"]=data_flat_avg_merged

