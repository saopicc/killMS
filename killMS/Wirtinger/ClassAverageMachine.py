from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from killMS.Other import ClassTimeIt
import scipy.ndimage

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
        T=ClassTimeIt.ClassTimeIt("AverageKernelMatrix")
        T.disable()
        #A0,A1=DicoData[""]
        A0=DicoData["A0"]
        A1=DicoData["A1"]
            
        NDir,Np,Npol=K.shape

        NpBlBlocks=DicoData["NpBlBlocks"][0]
        A0A1=sorted(list(set([(A0[i],A1[i]) for i in range(A0.size)])))
        
        NpOut=len(A0A1)
        NDirAvg=self.SM_Compress.NDir
        KOut=np.zeros((NDir,NDirAvg,NpOut,Npol),K.dtype)
        KOut0=np.zeros((NDir,NDirAvg,NpOut,Npol),K.dtype)
        
        IndList=[(np.where((A0==ThisA0)&(A1==ThisA1))[0]) for (ThisA0,ThisA1) in A0A1]

        f=DicoData["flags"]
        fp=f[:,:,0,0].copy()
        T.timeit("stuff")

        Labels=np.zeros((Np,Npol),np.int64)
        for iBl,ind in enumerate(IndList):
            Labels[ind,:]=iBl
            
        for iDirAvg in range(NDirAvg):
            K_Compress=self.PM_Compress.predictKernelPolCluster(DicoData,
                                                                self.SM_Compress,
                                                                iDirection=iDirAvg)
            T.timeit("K_Compress")
            k_rephase=K_Compress[:,:,0].conj()

            for iDir in range(NDir):
                p=K[iDir,:,:]#.copy()
                w=np.ones(p.shape,np.float64)
                w[p==0]=0.
                w[fp]=0.
                #p[fp]=0.

                #w.fill(1.)
                #print("!!")
                pp=w*p*k_rephase
                #pp.fill(1)
                #sw=

                Sr=scipy.ndimage.sum(pp.real,labels=Labels,index=np.arange(len(IndList)))
                Si=scipy.ndimage.sum(pp.imag,labels=Labels,index=np.arange(len(IndList)))
                Sw=scipy.ndimage.sum(w,labels=Labels,index=np.arange(len(IndList)))

                ind0=np.where(Sw>0)[0]
                if ind0.size==0: continue
                KOut0[iDir,iDirAvg,ind0,0]=(Sr+1j*Si)[ind0]/Sw[ind0]
                
                #if Sw.min()>0: stop
                
                # for iBl,ind in enumerate(IndList):
                #     # KOut[iDir,iDirAvg,iBl,0]=np.mean(pp[ind])
                #     sw=np.sum(w[ind,:])
                #     #print("0",iBl,sw)
                #     # print("!!")
                #     if sw==0: continue
                #     ppp=pp[ind,:]
                #     #print(pp[ind,:].size,sw)
                #     #print("1",iBl,np.sum(ppp))
                #     KOut[iDir,iDirAvg,iBl,0]=np.sum(ppp)/sw
                #     #KOut[iDir,iDirAvg,iBl,0]=sw
                # #print(KOut[iDir,iDirAvg].flat[:],KOut0[iDir,iDirAvg].flat[:],KOut[iDir,iDirAvg].flat[:]-KOut0[iDir,iDirAvg].flat[:])
                # #print(KOut[iDir,iDirAvg].flat[:]-KOut0[iDir,iDirAvg].flat[:])
                # #print("===============")
                # #if KOut[iDir,iDirAvg].max()>0.: stop
                
            T.timeit("Avg")
        #KOut0[np.isnan(KOut0)]=0.
        KOut=KOut0
        KOut=KOut.reshape((NDir,NDirAvg*NpOut,Npol))
     
        
        KOut[:,:,3]=KOut[:,:,0]
        n0,n1,_=KOut.shape

        # Mask=(KOut[:,:,0].reshape((n0,n1,1))==0)
        # _,n0,n1=Mask.shape
        # MaskMergeDir=np.ones((NDirAvg,1,1),Mask.dtype)*np.any(Mask,axis=0).reshape((1,n0,n1))
        # Mask=MaskMergeDir
        # KOut[:,:,0][Mask[:,:,0]]=0.
        # KOut[:,:,3][Mask[:,:,0]]=0.
        
        return KOut

    
    
    def AverageDataVector(self,DicoData,Mask=None,Stop=False,K=None,KCompress=None):
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
        
        DOut = np.zeros((NDirAvg,NpOut,self.NJacobBlocks_X,self.NJacobBlocks_Y),d.dtype)
        FOut=np.zeros(DOut.shape,f.dtype)

        # _,n0,n1=Mask.shape
        # MaskMergeDir=np.ones((NDirAvg,1,1),Mask.dtype)*np.any(Mask,axis=0).reshape((1,n0,n1))
        # Mask=MaskMergeDir

        Mask=np.any(K==0,axis=0)

#        stop
        
        for iDirAvg in range(NDirAvg):
            K_Compress=self.PM_Compress.predictKernelPolCluster(DicoData,self.SM_Compress,iDirection=iDirAvg)
            #print("!!")
            dp=d[:,:,0,0].copy()*K_Compress[:,:,0].conj()
            fp=f[:,:,0,0].copy()
            dp[Mask]=0.
            fp[Mask]=1
            for iBl,ind in enumerate(IndList):
                if np.min(fp[ind])==1:
                    #print("All flaged bl=%i, iDirAvg=%i"%(iBl,iDirAvg))
                    FOut[iDirAvg,iBl,0]=1
                    #stop
                    continue
                dps=dp[ind].ravel()
                fps=fp[ind].ravel()
                ws=np.float32((1-fps)).ravel()
                #ws[dps==0]=0.
                #print("!!")
                #ws.fill(1.)
                #ws[Mask[iDirAvg,ind,:].ravel()]=0.
                sws=np.sum(ws)
                if sws==0: continue
                # print("sws",ws.size,sws)
                # if ws.size!=sws: stop
                DOut[iDirAvg,iBl,0]=np.sum(dps*ws)/np.float32(sws)
                # if iBl==0:
                #     print("!!!!!?",iDirAvg)
                    
                # if Stop and iBl==0:
                #     import pylab
                #     S0=np.sum(dp[ind])
                #     S1=np.sum(K[0,ind,:])
                #     print("S0S1",np.abs(S0-S1))
                #     print("S0S1b",DOut[iDirAvg,iBl,0],)
                #     # pylab.clf()
                #     # pylab.subplot(1,2,1)
                #     # pylab.imshow(dp[ind].real,interpolation="nearest")
                #     # pylab.subplot(1,2,2)
                #     # pylab.imshow(K[iDirAvg,ind,:].real,interpolation="nearest")
                #     # pylab.suptitle("iBl=%i , iDir=%i, %s, %s"%(iBl,iDirAvg,S0,S1))
                #     # pylab.draw()
                #     # pylab.show(block=False)
                #     # pylab.pause(5)

                

                
        NChOut=1
        DicoData["flags_avg"]=FOut.reshape((NDirAvg*NpOut,NChOut,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        DicoData["data_avg"]=DOut.reshape((NDirAvg*NpOut,NChOut,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        
        DicoData["flags_flat_avg"]=np.rollaxis(FOut,2).reshape(self.NJacobBlocks_X,NDirAvg*NpOut*self.NJacobBlocks_Y)
        DicoData["data_flat_avg"]=np.rollaxis(DOut,2).reshape(self.NJacobBlocks_X,NDirAvg*NpOut*self.NJacobBlocks_Y)
        
        # print(DicoData["flags_avg"].shape,DicoData["data_avg"].shape,DicoData["flags_flat_avg"].shape,DicoData["data_flat_avg"].shape)
        # stop
        
        # if Stop:
        #     stop
        # print("Stop",Stop)
        
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
