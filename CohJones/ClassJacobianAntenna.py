import numpy as np
import NpShared
from PredictGaussPoints_NumExpr import ClassPredict

import ClassVisServer
import ClassSM
import ModLinAlg
import pylab

def testLM():
    SM=ClassSM.ClassSM("../TEST/ModelRandom00.txt.npy")
    rabeam,decbeam=SM.ClusterCat.ra,SM.ClusterCat.dec
    LofarBeam=("AE",5,rabeam,decbeam)
    VS=ClassVisServer.ClassVisServer("../TEST/0000.MS/")#,LofarBeam=LofarBeam)
    MS=VS.MS
    SM.Calc_LM(MS.rac,MS.decc)



    nd=SM.NDir
    npol=4
    na=MS.na
    Gains=np.zeros((na,nd,npol),dtype=np.complex64)
    Gains[:,:,0]=1
    Gains[:,:,-1]=1
    #Gains+=(np.random.randn(*Gains.shape)*0.5+1j*np.random.randn(*Gains.shape)
    #Gains=np.random.randn(*Gains.shape)+1j*np.random.randn(*Gains.shape)
    #GainsOrig=Gains.copy()
    ###############
    #GainsOrig=np.load("Rand.npz")["GainsOrig"]
    #Gains*=1e-3
    #Gains[:,0,:]=GainsOrig[:,0,:]
    ###############

    PolMode="HalfFull"
    # ### Scalar gains
    # PolMode="Scalar"
    # g=np.random.randn(*(Gains[:,:,0].shape))+1j*np.random.randn(*(Gains[:,:,0].shape))
    # g=g.reshape((na,nd,1))
    # Gains*=g
    # ####
    
    #Gains[:,2,:]=0
    #Gains[:,1,:]=0
    
    #Gains[:,1::,:]=0
    


    DATA=VS.GiveNextVis(0,1000)


    # Apply Jones
    PM=ClassPredict(Precision="S")
    #DATA["data"]=PM.predictKernelPolCluster(DATA,SM,ApplyJones=Gains)
    
    ############################
    iAnt=0
    JM=ClassJacobianAntenna(SM,iAnt,PolMode=PolMode)
    JM.setDATA(DATA)
    JM.CalcKernelMatrix()

    if PolMode=="Scalar":
        Gains=Gains[:,:,0].reshape((na,nd,1))
        G=Gains.copy().reshape((na,nd,1,1))
    else:
        G=Gains.copy().reshape((na,nd,2,2))
        

    y=JM.GiveDataVec()
    xtrue=JM.GiveSubVecGainAnt(Gains).flatten()
    x=Gains
    #################
    Radd=np.random.randn(*(G[iAnt].shape))#*0.3
    #np.savez("Rand",Radd=Radd,GainsOrig=GainsOrig)
    #################
    #Radd=np.load("Rand.npz")["Radd"]

    G[iAnt]+=Radd


    print "start"
    for i in range(10):
        xbef=G[iAnt].copy()
        x=JM.doLMStep(G)
        G[iAnt]=x
        
        pylab.figure(1)
        pylab.clf()
        pylab.plot(np.abs(xtrue.flatten()))
        pylab.plot(np.abs(x.flatten()))
        pylab.plot(np.abs(xtrue.flatten())-np.abs(x.flatten()))
        pylab.plot(np.abs(xbef.flatten()))
        pylab.ylim(-2,2)
        pylab.draw()
        pylab.show(False)

        # pylab.figure(3)
        # pylab.clf()
        # pylab.subplot(1,3,1)
        # pylab.imshow(np.abs(JM.Jacob)[0:20],interpolation="nearest")
        # pylab.subplot(1,3,2)
        # pylab.imshow(np.abs(JM.JHJ),interpolation="nearest")
        # pylab.subplot(1,3,3)
        # pylab.imshow(np.abs(JM.JHJinv),interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)

    stop    
    

import NpShared

class ClassJacobianAntenna():
    def __init__(self,SM,iAnt,PolMode="HalfFull",Lambda=1):
        self.PolMode=PolMode
        self.PM=ClassPredict(Precision="S")
        self.SM=SM
        self.iAnt=iAnt
        self.SharedDataDicoName="DicoData.%2.2i"%self.iAnt
        self.HasKernelMatrix=False
        self.Lambda=Lambda
    
    def GiveSubVecGainAnt(self,GainsIn):
        # if (GainsIn.size==self.NDir*2*2): return GainsIn.copy()
        Gains=GainsIn.copy().reshape((self.na,self.NDir,self.NJacobBlocks,self.NJacobBlocks))[self.iAnt]
        return Gains
        
    def setDATA(self,DATA):
        self.DATA=DATA
        
    def setDATA_Shared(self):
        # SharedNames=["SharedVis.freqs","SharedVis.times","SharedVis.A1","SharedVis.A0","SharedVis.flags","SharedVis.infos","SharedVis.uvw","SharedVis.data"]
        # self.DATA={}
        # for SharedName in SharedNames:
        #     key=SharedNames.split(".")[1]
        #     self.DATA[key]=NpShared.GiveArray(SharedName)
        self.DATA=NpShared.SharedToDico("SharedVis")
           
            
        
    def doLMStep(self,Gains):
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix()
        z=self.GiveDataVec()
        self.CalcJacobianAntenna(Gains)
        Ga=self.GiveSubVecGainAnt(Gains)
        Jx=self.J_x(Ga)
        zr=z-Jx

        # JH_z_0=np.load("LM.npz")["JH_z"]
        # x1_0=np.load("LM.npz")["x1"]
        # z_0=np.load("LM.npz")["z"]
        # Jx_0=np.load("LM.npz")["Jx"]

        JH_z=self.JH_z(zr)
        x1 = (1./(1.+self.Lambda)) * self.JHJinv_x(JH_z)
        
        

        # if self.iAnt==0:
        #     pylab.figure(2)
        #     pylab.clf()
        #     pylab.plot((z)[::11])
        #     pylab.plot((Jx)[::11])
        #     pylab.plot(zr[::11])
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)

        # pylab.figure(2)
        # pylab.clf()
        # #pylab.plot((z)[::11])
        # #pylab.plot((Jx-Jx_0)[::11])
        # #pylab.plot(zr[::11])
        # #pylab.plot(JH_z.flatten())
        # #pylab.plot(JH_z_0.flatten())
        # pylab.plot(x1.flatten())
        # pylab.plot(x1_0.flatten())
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        # stop
        # # np.savez("LM",JH_z=JH_z,x1=x1,z=z,Jx=Jx)
 
        # print JH_z.shape

        x0=Ga.flatten()
        x1+=x0
        return x1.reshape((self.NDir,self.NJacobBlocks,self.NJacobBlocks))

                                        
    def JHJinv_x(self,Gains):
        G=[]
        nd,_,_=Gains.shape
        for polIndex in range(self.NJacobBlocks):
            Gain=Gains[:,polIndex,:]
            Vec=np.dot(self.JHJinv,Gain.flatten())
            Vec=Vec.reshape((nd,1,self.NJacobBlocks))
            G.append(Vec)
            
        Gout=np.concatenate(G,axis=1)
        
        return Gout.flatten()


    def J_x(self,Gains):
        z=[]
        Jacob=self.Jacob
        Gains=Gains.reshape((self.NDir,self.NJacobBlocks,self.NJacobBlocks))
        for polIndex in range(self.NJacobBlocks):
            Gain=Gains[:,polIndex,:]
            z.append(np.dot(Jacob,Gain.flatten()))
        z=np.concatenate(z)
        return z

    def JH_z(self,zin):
        z=zin.reshape((self.NJacobBlocks,zin.size/self.NJacobBlocks))
        #z=zin.reshape((1,zin.size))
        Jacob=self.Jacob
        Gains=np.zeros((self.NDir,self.NJacobBlocks,self.NJacobBlocks),np.complex64)
        for polIndex in range(self.NJacobBlocks):
            ThisZ=z[polIndex]
            Gain=np.dot(Jacob.T.conj(),ThisZ.flatten())
            Gains[:,polIndex,:]=Gain.reshape((self.NDir,self.NJacobBlocks))

        return Gains

    def GiveDataVec(self):
        y=[]
        for polIndex in range(self.NJacobBlocks):
            y.append(self.Data[:,:,polIndex,:].flatten())
            
        y=np.concatenate(y)
        return y


    def CalcJacobianAntenna(self,GainsIn):
        if not(self.HasKernelMatrix): stop
        iAnt=self.iAnt
        NDir=self.NDir
        n4vis=self.n4vis
        na=self.na
        
        Gains=GainsIn.reshape((na,NDir,self.NJacobBlocks,self.NJacobBlocks))
        Jacob=np.zeros((n4vis,self.NJacobBlocks,NDir,self.NJacobBlocks),np.complex64)

        for iDir in range(NDir):
            G=Gains[self.A1,iDir].conj()

            K_XX=self.K_XX[iDir]
            K_YY=self.K_YY[iDir]

            nr=G.shape[0]
            J0=Jacob[:,0,iDir,0]
            g0_conj=G[:,0,0].reshape((nr,1))
            J0[:]=(g0_conj*K_XX).reshape((K_XX.size,))
            
            if self.PolMode=="HalfFull":
                J1=Jacob[:,0,iDir,1]
                J2=Jacob[:,1,iDir,0]
                J3=Jacob[:,1,iDir,1]

                g1_conj=G[:,1,0].reshape((nr,1))
                g2_conj=G[:,0,1].reshape((nr,1))
                g3_conj=G[:,1,1].reshape((nr,1))

                J1[:]=(g2_conj*K_YY).reshape((K_XX.size,))
                J2[:]=(g1_conj*K_XX).reshape((K_XX.size,))
                J3[:]=(g3_conj*K_YY).reshape((K_XX.size,))


        Jacob.shape=(n4vis*self.NJacobBlocks,NDir*self.NJacobBlocks)
        self.Jacob=Jacob
        J=Jacob
        self.JHJ=np.dot(J.T.conj(),J)
        self.JHJinv=ModLinAlg.invSVD(self.JHJ)
        # self.JHJinv=np.linalg.inv(self.JHJ)
        # self.JHJinv=np.diag(np.diag(self.JHJinv))

    def CalcKernelMatrix(self):
        # Out[28]: ['freqs', 'times', 'A1', 'A0', 'flags', 'uvw', 'data']
        DATA=self.DATA
        iAnt=self.iAnt
        na=DATA['infos'][0]
        self.na=na
        NDir=self.SM.NDir
        self.NDir=NDir
        self.iAnt=iAnt
        if self.PolMode=="HalfFull":
            npol=4

        self.DicoData=self.GiveData(DATA,iAnt)
        self.Data=self.DicoData["data"]
        self.A1=self.DicoData["A1"]
        # print "AntMax1",self.SharedDataDicoName,np.max(self.A1)
        # print self.DicoData["A1"]
        # print "AntMax0",self.SharedDataDicoName,np.max(self.DicoData["A0"])
        # print self.DicoData["A0"]
        nrows,nchan,_=self.Data.shape
        n4vis=nrows*nchan
        self.n4vis=n4vis
        
        KernelSharedName="KernelMat.%2.2i"%self.iAnt
        self.KernelMat=NpShared.GiveArray(KernelSharedName)
        if self.KernelMat!=None:
            self.HasKernelMatrix=True
            if self.PolMode=="HalfFull":
                self.K_XX=self.KernelMat[0]
                self.K_YY=self.KernelMat[1]
                self.NJacobBlocks=2
            elif self.PolMode=="Scalar":
                n4vis=self.Data.size
                self.K_XX=self.KernelMat[0]
                self.K_YY=self.K_XX
                self.n4vis=n4vis
                self.NJacobBlocks=1
            self.Data=self.Data.reshape((nrows,nchan,self.NJacobBlocks,self.NJacobBlocks))
            print "Kernel From shared"
            return
        else:
            print "COMPUTE KERNEL"

        # GiveArray(Name)

        if self.PolMode=="HalfFull":
            #self.K_XX=np.zeros((NDir,n4vis/nchan,nchan),np.complex64)
            #self.K_YY=np.zeros((NDir,n4vis/nchan,nchan),np.complex64)
            self.KernelMat=NpShared.zeros(KernelSharedName,(2,NDir,n4vis/nchan,nchan),dtype=np.complex64)
            self.K_XX=self.KernelMat[0]
            self.K_YY=self.KernelMat[1]
            # KernelMatrix=NpShared.zeros(KernelSharedName,(n4vis,NDir,2),dtype=np.complex64)
            self.NJacobBlocks=2
        elif self.PolMode=="Scalar":
            n4vis=self.Data.size
            # KernelMatrix_XX=np.zeros((NDir,n4vis,nchan),np.complex64)
            # KernelMatrix=NpShared.zeros(KernelSharedName,(n4vis,NDir,1),dtype=np.complex64)
            self.KernelMat=NpShared.zeros(KernelSharedName,(1,NDir,n4vis/nchan,nchan),dtype=np.complex64)
            self.K_XX=self.KernelMat[0]
            self.K_YY=self.K_XX
            self.n4vis=n4vis
            self.NJacobBlocks=1

        self.Data=self.Data.reshape((nrows,nchan,self.NJacobBlocks,self.NJacobBlocks))
            
        #self.K_XX=[]
        #self.K_YY=[]

        for iDir in range(NDir):
            K=self.PM.predictKernelPolCluster(self.DicoData,self.SM,iDirection=iDir)
            K_XX=K[:,:,0]
            K_YY=K[:,:,3]
            if self.PolMode=="Scalar":
                K_XX=(K_XX+K_YY)/2.
                K_YY=K_XX

            self.K_XX[iDir,:,:]=K_XX
            self.K_YY[iDir,:,:]=K_YY

            #self.K_XX.append(K_XX)
            #self.K_YY.append(K_YY)
        self.HasKernelMatrix=True


    def GiveData(self,DATA,iAnt):

        DicoData=NpShared.SharedToDico(self.SharedDataDicoName)
        if DicoData==False:
            print "COMPUTE DATA"
            DicoData={}
            ind0=np.where(DATA['A0']==iAnt)[0]
            ind1=np.where(DATA['A1']==iAnt)[0]
            DicoData["A0"] = np.concatenate([DATA['A0'][ind0], DATA['A1'][ind1]])
            DicoData["A1"] = np.concatenate([DATA['A1'][ind0], DATA['A0'][ind1]])
            D0=DATA['data'][ind0]
            D1=DATA['data'][ind1].conj()
            c1=D1[:,:,1].copy()
            c2=D1[:,:,2].copy()
            D1[:,:,1]=c2
            D1[:,:,2]=c1
            DicoData["data"] = np.concatenate([D0, D1])
            if self.PolMode=="Scalar":
                nr,nch,_=DicoData["data"].shape
                d=(DicoData["data"][:,:,0]+DicoData["data"][:,:,-1])/2
                DicoData["data"] = d.reshape((nr,nch,1))
            DicoData["uvw"]  = np.concatenate([DATA['uvw'][ind0], -DATA['uvw'][ind1]])
            DicoData["flags"] = np.concatenate([DATA['flags'][ind0], DATA['flags'][ind1]])
            DicoData["freqs"]   = DATA['freqs']
            DicoData["times"] = np.concatenate([DATA['times'][ind0], DATA['times'][ind1]])
            DicoData["infos"] = DATA['infos']
            DicoData=NpShared.DicoToShared(self.SharedDataDicoName,DicoData)
        else:
            print "DATA From shared"
            #print np.max(DicoData["A0"])
            #np.save("testA0",DicoData["A0"])
            #DicoData["A0"]=np.load("testA0.npy")
            #DicoData=NpShared.SharedToDico(self.SharedDataDicoName)
            #print np.max(DicoData["A0"])
            #print

            #stop

        if "DicoBeam" in DATA.keys():
            DicoData["DicoBeam"] = DATA["DicoBeam"]

        # DicoData["A0"] = np.concatenate([DATA['A0'][ind0]])
        # DicoData["A1"] = np.concatenate([DATA['A1'][ind0]])
        # D0=DATA['data'][ind0]
        # DicoData["data"] = np.concatenate([D0])
        # DicoData["uvw"]  = np.concatenate([DATA['uvw'][ind0]])
        # DicoData["flags"] = np.concatenate([DATA['flags'][ind0]])
        # DicoData["freqs"]   = DATA['freqs']


        return DicoData

###########################################
###########################################
###########################################


def testPredict():
    VS=ClassVisServer.ClassVisServer("../TEST/0000.MS/")

    MS=VS.MS
    SM=ClassSM.ClassSM("../TEST/ModelRandom00.txt.npy")
    SM.Calc_LM(MS.rac,MS.decc)




    nd=SM.NDir
    npol=4
    na=MS.na
    Gains=np.zeros((na,nd,npol),dtype=np.complex64)
    Gains[:,:,0]=1
    Gains[:,:,-1]=1
    #Gains+=np.random.randn(*Gains.shape)*0.5+1j*np.random.randn(*Gains.shape)
    Gains=np.random.randn(*Gains.shape)+1j*np.random.randn(*Gains.shape)
    #Gains[:,1,:]=0
    #Gains[:,2,:]=0
    #g=np.random.randn(*(Gains[:,:,0].shape))+1j*np.random.randn(*(Gains[:,:,0].shape))
    #g=g.reshape((na,nd,1))
    #Gains*=g

    DATA=VS.GiveNextVis(0,50)

    # Apply Jones
    PM=ClassPredict(Precision="S")
    DATA["data"]=PM.predictKernelPolCluster(DATA,SM,ApplyJones=Gains)
    
    ############################
    PolMode="HalfFull"#"Scalar"
    iAnt=10
    JM=ClassJacobianAntenna(SM,iAnt,PolMode=PolMode)
    JM.setDATA(DATA)
    JM.CalcKernelMatrix()
    if PolMode=="Scalar":
        Gains=Gains[:,:,0].reshape((na,nd,1))

    Jacob= JM.CalcJacobianAntenna(Gains)

    y=JM.GiveDataVec()
    
#    Gain=JM.ThisGain[:,1,:]
    predict=JM.J_x(Gains[iAnt])

    pylab.figure(1)
    pylab.clf()
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

    pylab.figure(2)
    pylab.clf()
    pylab.subplot(1,2,1)
    pylab.imshow(np.abs(JM.JHJ),interpolation="nearest")
    pylab.subplot(1,2,2)
    pylab.imshow(np.abs(JM.JHJinv),interpolation="nearest")
    pylab.draw()
    pylab.show(False)

    pylab.figure(3)
    pylab.clf()
    pylab.imshow(np.abs(JM.Jacob)[0:20],interpolation="nearest")
    pylab.draw()
    pylab.show(False)

    stop    
    
