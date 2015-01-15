import numpy as np
import NpShared
from PredictGaussPoints_NumExpr import ClassPredict

import ClassVisServer
import ClassSM
import ModLinAlg
import pylab
import ClassTimeIt

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
    


    DATA=VS.GiveNextVis()


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
    def __init__(self,SM,iAnt,PolMode="HalfFull",Lambda=1,Precision="S"):
        self.PolMode=PolMode
        #self.PM=ClassPredict(Precision="S")
        self.PM=ClassPredict(Precision=Precision)
        if Precision=="D":
            self.CType=np.complex128
            self.FType=np.float64
        if Precision=="S":
            self.CType=np.complex64
            self.FType=np.float32
            
        self.SM=SM
        self.iAnt=iAnt
        self.SharedDataDicoName="DicoData.%2.2i"%self.iAnt
        self.HasKernelMatrix=False
        self.Lambda=Lambda
        if self.PolMode=="HalfFull":
            self.NJacobBlocks=2
        elif self.PolMode=="Scalar":
            self.NJacobBlocks=1
    
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


    def PrepareJHJ(self,Pa,rms):
        Pinv=ModLinAlg.invSVD(Pa)
        self.JHJ=[]
        for ipol in range(self.NJacobBlocks):
            JHJ*=self.JHJ*(1./rms**2)
            JHJ+=Pinv
            JHJinv=ModLinAlg.invSVD(self.JHJ)
        #self.JHJinv=np.linalg.inv(self.JHJ)
        #print np.max(self.JHJinv1-self.JHJinv)
        #self.JHJinv=self.JHJinv1

    def AugmentPa(self,Pa_in):
        Pa=np.zeros((self.NDir*self.NJacobBlocks,self.NDir*self.NJacobBlocks),self.CType)
        for ipol in range(self.NJacobBlocks):
            for ipol in range(self.NJacobBlocks):
                Pa[ipol::self.NJacobBlocks,ipol::self.NJacobBlocks]=Pa_in[:,:]
        return Pa

    def ApplyK_vec(self,zr,rms,Pa):
        JH_z=self.JH_z(zr)


        # pylab.figure(1)
        # pylab.clf()
        # incr=1
        # pylab.subplot(2,1,1)
        # pylab.plot((z.flatten())[::incr].real)
        # pylab.plot((Jx.flatten())[::incr].real)
        # pylab.plot(zr.flatten()[::incr].real)
        # pylab.subplot(2,1,2)
        # pylab.plot((z.flatten())[::incr].imag)
        # pylab.plot((Jx.flatten())[::incr].imag)
        # pylab.plot(zr.flatten()[::incr].imag)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop


        # pylab.figure(1)
        # pylab.clf()
        # incr=1
        # pylab.subplot(2,1,1)
        # pylab.imshow(np.abs(self.JHJ))
        # pylab.subplot(2,1,2)
        # pylab.imshow(np.abs(self.JHJinv))
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop
        



        x1 = self.JHJinv_x(JH_z)
        z1=self.J_x(x1)/rms**2
        zr-=z1
        x2=self.JH_z(zr)
        x3=[]
        for ipol in range(self.NJacobBlocks):
            x3.append(np.dot(Pa,x2[:,ipol,:].flatten())/rms**2)
        x3=np.concatenate(x3,axis=1)

        return x3
        
    def doEKFStep(self,Gains,P,rms):
        T=ClassTimeIt.ClassTimeIt("EKF")
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix()
            T.timeit("CalcKernelMatrix")
        z=self.DicoData["data_flat"]#self.GiveDataVec()
        self.CalcJacobianAntenna(Gains)
        T.timeit("Jacob")

        Ga=self.GiveSubVecGainAnt(Gains)

        Jx=self.J_x(Ga)
        T.timeit("J_x")


        
        Pa=P[self.iAnt]
        #Pa=self.AugmentPa(Pa)
        T.timeit("AugmentPa")

        self.PrepareJHJ(Pa,rms)
        T.timeit("PrepareJHJ")
        #rms=self.DATA["rms"]

        zr=(z-Jx)
        T.timeit("zr")
        x3=self.ApplyK_vec(zr,rms,Pa)
        T.timeit("ApplyK_vec")
        x0=Ga.flatten()
        x4=x3+x0

        npars=Pa.shape[0]
        T.timeit("Add")
        Pa_new=np.zeros_like(Pa)
        for iPar in range(npars):
            J_Px=self.J_x(Pa[:,iPar])
            xP=self.ApplyK_vec(J_Px,rms,Pa)
            Pa_new[:,iPar]=xP
        T.timeit("Pa")
        Pa_new= Pa-Pa_new#(np.diag(np.diag(Pa-Pa_new)))#Pa-Pa_new#np.abs(np.diag(np.diag(Pa-Pa_new)))
        Pa_new+=np.diag(np.ones((Pa_new.shape[0],)))*0.005**2
        #Pa_new2=np.abs(Pa_new)
        del(self.Jacob,self.JHJ,self.JHJinv)

        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.plot(Pa)
        # pylab.subplot(1,2,2)
        # pylab.plot(Pa_new)
        # pylab.draw()
        # pylab.show()

        #pylab.plot(Pa_new2)
        


        return x4.reshape((self.NDir,self.NJacobBlocks,self.NJacobBlocks)),Pa_new
           
            
        
    def doLMStep(self,Gains):
        #print
        T=ClassTimeIt.ClassTimeIt("doLMStep")
        T.disable()
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix()
            T.timeit("CalcKernelMatrix")
        z=self.DicoData["data_flat"]#self.GiveDataVec()
        self.CalcJacobianAntenna(Gains)
        T.timeit("CalcJacobianAntenna")
        Ga=self.GiveSubVecGainAnt(Gains)
        T.timeit("GiveSubVecGainAnt")
        Jx=self.J_x(Ga)
        T.timeit("Jx")
        zr=z-Jx
        T.timeit("resid")

        # JH_z_0=np.load("LM.npz")["JH_z"]
        # x1_0=np.load("LM.npz")["x1"]
        # z_0=np.load("LM.npz")["z"]
        # Jx_0=np.load("LM.npz")["Jx"]

        JH_z=self.JH_z(zr)
        T.timeit("JH_z")
        #self.JHJinv=ModLinAlg.invSVD(self.JHJ)
        self.JHJinv=np.linalg.inv(self.JHJ)
        x1 = (1./(1.+self.Lambda)) * self.JHJinv_x(JH_z)
        T.timeit("self.JHJinv_x")
        
        
        
        # if self.iAnt==5:
        #     f=(self.DicoData["flags_flat"]==0)
        #     pylab.figure(2)
        #     pylab.clf()
        #     pylab.plot((z[f]))#[::11])
        #     pylab.plot((Jx[f]))#[::11])
        #     pylab.plot(zr[f])#[::11])
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     stop
        # # pylab.figure(2)
        # # pylab.clf()
        # # #pylab.plot((z)[::11])
        # # #pylab.plot((Jx-Jx_0)[::11])
        # # #pylab.plot(zr[::11])
        # # #pylab.plot(JH_z.flatten())
        # # #pylab.plot(JH_z_0.flatten())
        # # pylab.plot(x1.flatten())
        # # pylab.plot(x1_0.flatten())
        # # pylab.draw()
        # # pylab.show(False)
        # # pylab.pause(0.1)

        # stop
        # # np.savez("LM",JH_z=JH_z,x1=x1,z=z,Jx=Jx)
 
        # print JH_z.shape

        x0=Ga.flatten()
        x1+=x0
        del(self.Jacob)
        T.timeit("rest")

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
            
            Gain=Gains[:,polIndex,:].flatten()

            #flags=self.DicoData["flags_flat"][polIndex]
            J=Jacob#[flags==0]
            z.append(np.dot(J,Gain))
        z=np.array(z)
        return z

    def JH_z(self,zin):
        #z=zin.reshape((self.NJacobBlocks,zin.size/self.NJacobBlocks))
        #z=zin.reshape((1,zin.size))
        Jacob=self.Jacob
        Gains=np.zeros((self.NDir,self.NJacobBlocks,self.NJacobBlocks),self.CType)
        for polIndex in range(self.NJacobBlocks):
            
            flags=self.DicoData["flags_flat"][polIndex]
            ThisZ=zin[polIndex][flags==0]#self.DicoData["flags_flat"[polIndex]
            
            J=Jacob[flags==0]

            Gain=np.dot(J.T.conj(),ThisZ.flatten())
            Gains[:,polIndex,:]=Gain.reshape((self.NDir,self.NJacobBlocks))

        return Gains

    # def GiveDataVec(self):
    #     y=[]
    #     yf=[]
    #     for polIndex in range(self.NJacobBlocks):
    #         DataVec=self.DicoData["data"][:,:,polIndex,:].flatten()
    #         Flags=self.DicoData["flags"][:,:,polIndex,:].flatten()

    #         y.append(DataVec)
    #         yf.append(Flags)

    #     y=np.concatenate(y)
    #     yf=np.concatenate(yf)
    #     return y,yf


    def CalcJacobianAntenna(self,GainsIn):
        if not(self.HasKernelMatrix): stop
        iAnt=self.iAnt
        NDir=self.NDir
        n4vis=self.n4vis
        na=self.na
        
        Gains=GainsIn.reshape((na,NDir,self.NJacobBlocks,self.NJacobBlocks))
        Jacob=np.zeros((n4vis,self.NJacobBlocks,NDir,self.NJacobBlocks),self.CType)

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
        # self.Data=self.DicoData["data"]
        self.A1=self.DicoData["A1"]
        # print "AntMax1",self.SharedDataDicoName,np.max(self.A1)
        # print self.DicoData["A1"]
        # print "AntMax0",self.SharedDataDicoName,np.max(self.DicoData["A0"])
        # print self.DicoData["A0"]
        nrows,nchan,_,_=self.DicoData["flags"].shape
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
                #n4vis=self.DicoData["data_flat"].size
                self.K_XX=self.KernelMat[0]
                self.K_YY=self.K_XX
                self.n4vis=n4vis
                self.NJacobBlocks=1
            # self.Data=self.Data.reshape((nrows,nchan,self.NJacobBlocks,self.NJacobBlocks))

            #print "Kernel From shared"
            return
        else:
            pass
            #print "COMPUTE KERNEL"

        # GiveArray(Name)

        if self.PolMode=="HalfFull":
            #self.K_XX=np.zeros((NDir,n4vis/nchan,nchan),np.complex64)
            #self.K_YY=np.zeros((NDir,n4vis/nchan,nchan),np.complex64)
            self.KernelMat=NpShared.zeros(KernelSharedName,(2,NDir,n4vis/nchan,nchan),dtype=self.CType)
            self.K_XX=self.KernelMat[0]
            self.K_YY=self.KernelMat[1]
            # KernelMatrix=NpShared.zeros(KernelSharedName,(n4vis,NDir,2),dtype=np.complex64)
            self.NJacobBlocks=2
        elif self.PolMode=="Scalar":
            #n4vis=self.Data.size
            n4vis=self.DicoData["data_flat"].size
            # KernelMatrix_XX=np.zeros((NDir,n4vis,nchan),np.complex64)
            # KernelMatrix=NpShared.zeros(KernelSharedName,(n4vis,NDir,1),dtype=np.complex64)
            self.KernelMat=NpShared.zeros(KernelSharedName,(1,NDir,n4vis/nchan,nchan),dtype=self.CType)
            self.K_XX=self.KernelMat[0]
            self.K_YY=self.K_XX
            self.n4vis=n4vis
            self.NJacobBlocks=1

        #self.Data=self.Data.reshape((nrows,nchan,self.NJacobBlocks,self.NJacobBlocks))

        #self.K_XX=[]
        #self.K_YY=[]

        ApplyTimeJones=None
        if "DicoBeam" in self.DicoData.keys():
            ApplyTimeJones=self.DicoData["DicoBeam"]

        for iDir in range(NDir):
            
            K=self.PM.predictKernelPolCluster(self.DicoData,self.SM,iDirection=iDir,ApplyTimeJones=ApplyTimeJones)
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
            #print "COMPUTE DATA"
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
            DicoData["uvw"]  = np.concatenate([DATA['uvw'][ind0], -DATA['uvw'][ind1]])

            D0=DATA['flags'][ind0]
            D1=DATA['flags'][ind1].conj()
            c1=D1[:,:,1].copy()
            c2=D1[:,:,2].copy()
            D1[:,:,1]=c2
            D1[:,:,2]=c1
            DicoData["flags"] = np.concatenate([D0, D1])

            if self.PolMode=="Scalar":
                nr,nch,_=DicoData["data"].shape
                d=(DicoData["data"][:,:,0]+DicoData["data"][:,:,-1])/2
                DicoData["data"] = d.reshape((nr,nch,1))

                f=(DicoData["flags"][:,:,0]|DicoData["flags"][:,:,-1])
                DicoData["flags"] = f.reshape((nr,nch,1))


            DicoData["freqs"]   = DATA['freqs']
            DicoData["times"] = np.concatenate([DATA['times'][ind0], DATA['times'][ind1]])
            DicoData["infos"] = DATA['infos']

            nr,nch,_=DicoData["data"].shape
            DicoData["flags"]=DicoData["flags"].reshape(nr,nch,self.NJacobBlocks,self.NJacobBlocks)
            DicoData["data"]=DicoData["data"].reshape(nr,nch,self.NJacobBlocks,self.NJacobBlocks)

            DicoData["flags_flat"]=np.rollaxis(DicoData["flags"],2).reshape(self.NJacobBlocks,nr*nch*self.NJacobBlocks)
            DicoData["data_flat"]=np.rollaxis(DicoData["data"],2).reshape(self.NJacobBlocks,nr*nch*self.NJacobBlocks)
            #DicoData["data_flat"]=DicoData["data_flat"][DicoData["flags_flat"]==0]

            del(DicoData["data"])

            DicoData=NpShared.DicoToShared(self.SharedDataDicoName,DicoData)
        else:
            pass
            #print "DATA From shared"
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
    
