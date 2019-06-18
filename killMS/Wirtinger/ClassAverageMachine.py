import numpy as np

class ClassAverageMachine():
    def __init__(self,GD,PM_Compress,SM_Compress,PolMode="Scalar"):
        self.GD=GD
        self.PM_Compress=PM_Compress
        self.SM_Compress=SM_Compress
        self.PolMode=PolMode
        if self.GD["Solvers"]["NChanSols"]!=0:
            raise ValueError("Only deal with NChanSol=0")
        if self.GD["Solvers"]["PolMode"]!="Scalar":
            raise ValueError("Only deal with PolMode=Scalar")

    def AverageKernelMatrix(self,DicoData,K):
        #A0,A1=DicoData[""]
        A0=DicoData["A0_freq_flat"][0]
        A1=DicoData["A1_freq_flat"][0]
            
        NDir,Np,Npol=K.shape

        A0A1=sorted(list(set([(A0[i],A1[i]) for i in range(Np)])))
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
