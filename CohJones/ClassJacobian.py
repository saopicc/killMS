import numpy as np
import NpShared
from PredictGaussPoints_NumExpr import ClassPredict

import ClassVisServer
import ClassSM


def test():
    VS=ClassVisServer.ClassVisServer("TEST/0000.MS/")

    MS=VS.MS
    SM=ClassSM.ClassSM("TEST/ModelRandom00.txt.npy")
    SM.Calc_LM(MS.rac,MS.decc)




    nd=SM.NDir
    npol=4
    na=MS.na
    Gains=np.zeros((na,nd,npol),dtype=np.complex64)
    Gains[:,:,0]=1
    Gains[:,:,-1]=1
    Gains+=np.random.randn(*Gains.shape)*0.5+1j*np.random.randn(*Gains.shape)
    DATA=VS.GiveNextVis(0,50)

    # Apply Jones
    PM=ClassPredict(Precision="S")
    DATA["data"]=PM.predictKernelPolCluster(DATA,SM,ApplyJones=Gains)
    
    JM=ClassJacobian(SM)
    Jacob= JM.GiveJacobianAntenna(DATA,Gains,15)[0][1]

    y=JM.Data[:,:,0,:].flatten()

    Gain=JM.ThisGain[:,0,:]
    predict=np.dot(Jacob,Gain.flatten())

    import pylab
    pylab.clf()
    #pylab.plot(Jacob.T)

    pylab.subplot(2,1,1)
    pylab.plot(predict.real)
    pylab.plot(y.real)
    pylab.plot((predict-y).real)
    pylab.subplot(2,1,2)
    pylab.plot(predict.imag)
    pylab.plot(y.imag)
    pylab.plot((predict-y).imag)
    pylab.draw()
    pylab.show(False)
    stop    
    



class ClassJacobian():
    def __init__(self,SM,PolMode="HalfFull"):
        self.PolMode=PolMode
        self.PM=ClassPredict(Precision="S")
        self.SM=SM

    def GiveJacobianAntenna(self,DATA,Gains,iAnt):
        # Out[28]: ['freqs', 'times', 'A1', 'A0', 'flags', 'uvw', 'data']

        na=DATA['infos'][0]
        NDir=self.SM.NDir
        if self.PolMode=="HalfFull":
            npol=4
        Gains=Gains.copy().reshape((na,NDir,2,2))

        ThisGain=Gains[iAnt].copy()
        self.ThisGain=ThisGain

        ListDicoData=[self.GiveData(DATA,iAnt,False), self.GiveData(DATA,iAnt,True)]
        ListData=[ThisDATA["data"] for ThisDATA in ListDicoData]
        self.Data=np.concatenate(ListData)
        ListA1=[ThisDATA["A1"] for ThisDATA in ListDicoData]
        self.A1=np.concatenate(ListA1)

        nrows,nchan,_=self.Data.shape
        self.Data=self.Data.reshape((nrows,nchan,2,2))


        if self.PolMode=="HalfFull":
            n4vis=self.Data.size/4
            JacobList=[[iPolBlock, np.zeros((n4vis,2,NDir,2),np.complex64)] for iPolBlock in range(1)]
        
        for iPolBlock,Jacob in JacobList:
            for iDir in range(NDir):
                K=np.concatenate([self.GiveKernel(DATA,iAnt,iDir,False), self.GiveKernel(DATA,iAnt,iDir,True)])
                K_XX=K[:,:,0]
                K_YY=K[:,:,3]
                J0=Jacob[:,0,iDir,0]
                J1=Jacob[:,0,iDir,1]
                J2=Jacob[:,1,iDir,0]
                J3=Jacob[:,1,iDir,1]
                
                

                G=Gains[self.A1,iDir].conj()

                nr=G.shape[0]
                g0_conj=G[:,0,0].reshape((nr,1))
                g1_conj=G[:,1,0].reshape((nr,1))
                g2_conj=G[:,0,1].reshape((nr,1))
                g3_conj=G[:,1,1].reshape((nr,1))

                J0[:]=(g0_conj*K_XX).reshape((K_XX.size,))
                J1[:]=(g2_conj*K_YY).reshape((K_XX.size,))
                J2[:]=(g1_conj*K_XX).reshape((K_XX.size,))
                J3[:]=(g3_conj*K_YY).reshape((K_XX.size,))
            Jacob.shape=(n4vis*2,NDir*2)

        self.JacobList=JacobList
        return JacobList

        


    def GiveData(self,DATA,iAnt,revert):
        DicoData={}
        if not(revert):
            ind=np.where(DATA['A0']==iAnt)[0]
            DicoData["A0"]  = DATA['A0'][ind]
            DicoData["A1"]  = DATA['A1'][ind]
            DicoData["data"]  = DATA['data'][ind]
            DicoData["uvw"]   = DATA['uvw'][ind]
        else:
            ind=np.where(DATA['A1']==iAnt)[0]
            DicoData["A1"]  = DATA['A0'][ind]
            DicoData["A0"]  = DATA['A1'][ind]
            DicoData["data"]  = DATA['data'][ind].conj()
            DicoData["uvw"]   = -DATA['uvw'][ind]
            
        DicoData["flags"] = DATA['flags'][ind]
        DicoData["freqs"]   = DATA['freqs']

        return DicoData

    def GiveKernel(self,DATA,iAnt,iDir,revert):
        DicoData={}
        if not(revert):
            ind=np.where(DATA['A0']==iAnt)[0]
            DicoData["A0"]  = DATA['A0'][ind]
            DicoData["A1"]  = DATA['A1'][ind]
            DicoData["uvw"]   = DATA['uvw'][ind]
        else:
            ind=np.where(DATA['A1']==iAnt)[0]
            DicoData["A1"]  = DATA['A0'][ind]
            DicoData["A0"]  = DATA['A1'][ind]
            DicoData["uvw"]   = -DATA['uvw'][ind]
            

        DicoData["flags"] = DATA['flags'][ind]
        DicoData["freqs"]   = DATA['freqs']
        
        kernel=self.PM.predictKernelPolCluster(DicoData,self.SM,iDirection=iDir)

        return kernel

            

