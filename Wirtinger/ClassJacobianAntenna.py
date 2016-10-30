import numpy as np
from killMS2.Array import NpShared
from killMS2.Predict.PredictGaussPoints_NumExpr5 import ClassPredict
import os
from killMS2.Data import ClassVisServer
#from Sky import ClassSM
from killMS2.Array import ModLinAlg
import pylab
from killMS2.Other import ClassTimeIt
from killMS2.Array.Dot import NpDotSSE

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

    PolMode="IFull"
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
    



class ClassJacobianAntenna():
    def __init__(self,SM,iAnt,PolMode="IFull",Precision="S",PrecisionDot="D",IdSharedMem="",
                 PM=None,GD=None,NChanSols=1,ChanSel=None,
                 SharedDicoDescriptors=None,
                 **kwargs):
        T=ClassTimeIt.ClassTimeIt("  InitClassJacobianAntenna")
        T.disable()

        self.ChanSel=ChanSel
        self.SharedDicoDescriptors=SharedDicoDescriptors
        self.GD=GD
        self.IdSharedMem=IdSharedMem
        self.PolMode=PolMode
        #self.PM=ClassPredict(Precision="S")
        self.Rinv_flat=None
        for key in kwargs.keys():
            setattr(self,key,kwargs[key])
        self.PM=PM
        self.SM=SM
        T.timeit("Init0")
        if PM==None:
            self.PM=ClassPredict(Precision=Precision,
                                 DoSmearing=self.DoSmearing,
                                 IdMemShared=IdSharedMem)
            
            if self.GD["ImageSkyModel"]["BaseImageName"]!="":
                self.PM.InitGM(self.SM)

        T.timeit("PM")
        if PrecisionDot=="D":
            self.CType=np.complex128
            self.FType=np.float64

        if PrecisionDot=="S":
            self.CType=np.complex64
            self.FType=np.float32

        self.CType=np.complex128
        self.TypeDot="Numpy"
        #self.TypeDot="SSE"

        self.iAnt=int(iAnt)
        self.SharedDataDicoName="%sDicoData.%2.2i"%(self.IdSharedMem,self.iAnt)
        self.NChanSols=NChanSols
        
        if self.PolMode=="IFull":
            self.NJacobBlocks_X=2
            self.NJacobBlocks_Y=2
            self.npolData=4
        
        elif self.PolMode=="Scalar":
            self.NJacobBlocks_X=1
            self.NJacobBlocks_Y=1
            self.npolData=1
        
        elif self.PolMode=="IDiag":
            self.NJacobBlocks_X=2
            self.NJacobBlocks_Y=1
            self.npolData=2
        

        self.Reinit()
        T.timeit("rest")

    def Reinit(self):
        self.HasKernelMatrix=False
        self.LQxInv=None

    def GiveSubVecGainAnt(self,GainsIn):
        # if (GainsIn.size==self.NDir*2*2): return GainsIn.copy()
        Gains=GainsIn.copy().reshape((self.na,self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))[self.iAnt]
        return Gains
        
    def setDATA(self,DATA):
        self.DATA=DATA
        
    def setDATA_Shared(self):
        # SharedNames=["SharedVis.freqs","SharedVis.times","SharedVis.A1","SharedVis.A0","SharedVis.flags","SharedVis.infos","SharedVis.uvw","SharedVis.data"]
        # self.DATA={}
        # for SharedName in SharedNames:
        #     key=SharedNames.split(".")[1]
        #     self.DATA[key]=NpShared.GiveArray(SharedName)

        T=ClassTimeIt.ClassTimeIt("  setDATA_Shared")
        T.disable()
        
        #self.DATA=NpShared.SharedToDico("%sSharedVis"%self.IdSharedMem)
        self.DATA=NpShared.SharedObjectToDico(self.SharedDicoDescriptors["SharedVis"])



        _,self.NChanMS,_=self.DATA["data"].shape
        if self.ChanSel==None:
            self.ch0=0
            self.ch1=self.NChanMS
        else:
            self.ch0,self.ch1=self.ChanSel

        
        self.NChanData=self.ch1-self.ch0

        T.timeit("SharedToDico0")
        #DicoBeam=NpShared.SharedToDico("%sPreApplyJones"%self.IdSharedMem)
        DicoBeam=NpShared.SharedObjectToDico(self.SharedDicoDescriptors["PreApplyJones"])
        
        T.timeit("SharedToDico1")
        if DicoBeam!=None:
            self.DATA["DicoPreApplyJones"]=DicoBeam
            # self.DATA["DicoClusterDirs"]=NpShared.SharedToDico("%sDicoClusterDirs"%self.IdSharedMem)
            self.DATA["DicoClusterDirs"]=NpShared.SharedObjectToDico(self.SharedDicoDescriptors["DicoClusterDirs"])


        T.timeit("SharedToDico2")


        #self.DATA["UVW_RefAnt"]=NpShared.GiveArray("%sUVW_RefAnt"%self.IdSharedMem)

    def GivePaPol(self,Pa_in,ipol):
        PaPol=Pa_in.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y,self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        PaPol=PaPol[:,ipol,:,:,ipol,:].reshape((self.NDir*self.NJacobBlocks_Y,self.NDir*self.NJacobBlocks_Y))
        return PaPol

    def setQxInvPol(self):
        QxInv=np.ones((self.NDir,1,self.NJacobBlocks_Y,self.NDir,1,self.NJacobBlocks_Y),np.float32)
        QxInv*=1./(self.AmpQx**2)
        QxInv=QxInv.reshape((self.NDir*self.NJacobBlocks_Y,self.NDir*self.NJacobBlocks_Y))
        self.LQxInv=[QxInv,QxInv]

    def PrepareJHJ_EKF(self,Pa_in,rms):
        self.L_JHJinv=[]
        incr=1
        # pylab.figure(1)
        # pylab.clf()
        # pylab.imshow(np.abs(self.JHJ),interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        for ipol in range(self.NJacobBlocks_X):
            PaPol=self.GivePaPol(Pa_in,ipol)
            Pinv=ModLinAlg.invSVD(PaPol)
            JHJ=self.L_JHJ[ipol]#*(1./rms**2)
            JHJ+=Pinv
            if self.DoReg:
                JHJ+=self.LQxInv[ipol]*(self.gamma**2)
            JHJinv=ModLinAlg.invSVD(JHJ)
            self.L_JHJinv.append(JHJinv)

    def CalcKapa_i(self,yr,Pa,rms):
        kapaout=0
        for ipol in range(self.NJacobBlocks_X):
            J=self.LJacob[ipol]
            PaPol=self.GivePaPol(Pa,ipol)
            pa=np.abs(np.diag(PaPol))
            pa=pa.reshape(1,pa.size)
            JP=J*pa
            trJPJH=np.sum(np.abs(JP*J.conj()))
            trYYH=np.sum(np.abs(yr)**2)
            Np=np.where(self.DicoData["flags_flat"]==0)[0].size
            Take=(self.DicoData["flags_flat"]==0)
            trR=np.sum(self.R_flat[Take])#Np*rms**2
            kapa=np.abs((trYYH-trR)/trJPJH)
            kapaout+=np.sqrt(kapa)
            # print self.iAnt,rms,np.sqrt(kapa),trYYH,trR,trJPJH,pa
        kapaout=np.max([1.,kapaout])
        return kapaout

    def PrepareJHJ_LM(self):

        self.L_JHJinv=[]
        if self.DataAllFlagged:
            return

        for ipol in range(self.NJacobBlocks_X):

            M=self.L_JHJ[ipol]
            if self.DoTikhonov:
                self.LambdaTkNorm=self.LambdaTk*np.mean(np.abs(np.diag(M)))
                
                # Lin.shape= (self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)
                Linv=np.diag(self.Linv[:,ipol,:].ravel())
                Linv*=self.LambdaTkNorm/(1.+self.LambdaLM)
                M2=M+Linv

                # pylab.clf()
                # pylab.subplot(1,2,1)
                # pylab.imshow(np.abs(M),interpolation="nearest")
                # pylab.colorbar()
                # pylab.subplot(1,2,2)
                # pylab.imshow(np.abs(Linv),interpolation="nearest")
                # pylab.colorbar()
                # pylab.draw()
                # pylab.show(False)
                # pylab.pause(0.1)
                # stop

            else:
                M2=M


            JHJinv=ModLinAlg.invSVD(M2)
            #JHJinv=ModLinAlg.invSVD(self.JHJ)
            self.L_JHJinv.append(JHJinv)


    def ApplyK_vec(self,zr,rms,Pa,DoReg=True):

        

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


        # if self.iAnt==5:
        #     pylab.figure(1)
        #     pylab.clf()
        #     incr=1
        #     f=(self.DicoData["flags_flat"]==0)
        #     pylab.subplot(2,1,1)
        #     pylab.plot((z1[f].flatten())[::incr].real)
        #     pylab.plot((zr[f].flatten())[::incr].real)
        #     pylab.plot((z1-zr)[f].flatten()[::incr].real)
        #     pylab.subplot(2,1,2)
        #     pylab.plot((z1[f].flatten())[::incr].imag)
        #     pylab.plot((zr[f].flatten())[::incr].imag)
        #     pylab.plot((z1-zr)[f].flatten()[::incr].imag)
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)

        Rinv_zr=self.Rinv_flat*zr
        JH_z=self.JH_z(Rinv_zr)

        x1 = self.JHJinv_x(JH_z)
        if (self.DoReg)&(DoReg):
            self.G0=np.ones_like(self.Ga)#*200.
            dx1a=self.Msq_x(self.LQxInv,(self.Ga-self.G0))
            dx1b = self.gamma*self.JHJinv_x(dx1a)
            print "x'_0:",dx1b
            x1+=dx1b
            print "x':",x1

        z1 = self.J_x(x1)

        zr-=z1
        zr*=self.Rinv_flat
        x2=self.JH_z(zr)

        if (self.DoReg)&(DoReg):
            xr=self.Ga.ravel()-self.G0.ravel()-self.gamma*x1.ravel()
            dx2=self.gamma*self.Msq_x(self.LQxInv,xr)
            x2+=dx2.reshape(x2.shape)

        x3=[]
        for ipol in range(self.NJacobBlocks_X):
            PaPol=self.GivePaPol(Pa,ipol)
            #print PaPol,PaPol.shape


            if self.TypeDot=="Numpy":
                Prod=np.dot(PaPol,x2[:,ipol,:].flatten())
            elif self.TypeDot=="SSE":
                X2=x2[:,ipol,:].flatten()
                X2=X2.reshape((1,X2.size))
                Prod=NpDotSSE.dot_A_BT(PaPol.copy(),X2)
            

            x3.append(Prod.reshape((self.NDir,1,self.NJacobBlocks_Y)))

            

        x3=np.concatenate(x3,axis=1)
        #x3=np.swapaxes(x3,1,2)

        return x3
    

        
    def doEKFStep(self,Gains,P,evP,rms,Gains0Iter=None):
        T=ClassTimeIt.ClassTimeIt("    EKF")
        T.disable()
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix(rms)
            self.SelectChannelKernelMat()
            T.timeit("CalcKernelMatrix")
        z=self.DicoData["data_flat"]#self.GiveDataVec()

        f=(self.DicoData["flags_flat"]==0)
        ind=np.where(f)[0]
        Pa=P[self.iAnt]
        Ga=self.GiveSubVecGainAnt(Gains)
        self.Ga=Ga
        self.rms=rms


        self.rmsFromData=None
        if ind.size==0:
            return Ga.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),Pa,{"std":-1.,"max":-1.,"kapa":-1.}
        if self.DoReg:
            self.setQxInvPol()
        
        self.CalcJacobianAntenna(Gains)
        T.timeit("Jacob")
        
        # if Gains0Iter!=None:
        #     Ga=self.GiveSubVecGainAnt(Gains0Iter)

        Jx=self.J_x(Ga)
        T.timeit("J_x")

        

        self.PrepareJHJ_EKF(Pa,rms)
        T.timeit("PrepareJHJ")

        # estimate x
        zr=(z-Jx)
        zr[self.DicoData["flags_flat"]]=0

        T.timeit("Resid")

        kapa=self.CalcKapa_i(zr,Pa,rms)

        InfoNoise={"std":np.std(zr[f]),"max":np.max(np.abs(zr[f])),"kapa":kapa}
        #print self.iAnt,InfoNoise
        #T.timeit("kapa")

        self.rmsFromData=np.std(zr[f])
        T.timeit("rmsFromData")

        # if np.isnan(self.rmsFromData):
        #     print zr
        #     print zr[f]
        #     print self.rmsFromData
        #     stop

        # if self.iAnt==51:
        #     #self.DicoData["flags_flat"].fill(0)
        #     f=(self.DicoData["flags_flat"]==0)
        #     fig=pylab.figure(2)
        #     pylab.clf()
        #     pylab.plot((z[f]))#[::11])#[::11])
        #     pylab.plot((Jx[f]))#[::11])#[::11])

        #     #pylab.plot(zr[f])#[::11])#[::11])
        #     #pylab.draw()
        #     ifile=0
        #     while True:
        #         fname="png/png.%5.5i.png"%ifile
        #         if os.path.isfile(fname) :
        #             fig.savefig(fname)
        #             break
        #         ifile+=1
                
                
        #     #pylab.show(False)
        #     #pylab.pause(0.1)
        #     #stop


        x3=self.ApplyK_vec(zr,rms,Pa)

        T.timeit("ApplyK_vec")
        x0=Ga.flatten()
        x4=x0+self.LambdaKF*x3.flatten()

        # estimate P


        #Pa_new1=Pa-np.dot(evPa,Pa)
        evPa=evP[self.iAnt]
        Pa_new1=np.dot(evPa,Pa)
        #Pa_new1=Pa


        T.timeit("EstimateP")
        # ##################
        # for iPar in range(Pa.shape[0]):
        #     J_Px=self.J_x(Pa[iPar,:])
        #     xP=self.ApplyK_vec(J_Px,rms,Pa)
        #     evPa[iPar,:]=xP.flatten()
        # evPa= Pa-evPa
        # Pa_new1=evPa

        del(self.LJacob)
        T.timeit("Rest")
        
        return x4.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),Pa_new1,InfoNoise


    def CalcMatrixEvolveCov(self,Gains,P,rms):
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix(rms)
            self.SelectChannelKernelMat()
        if self.LQxInv==None:
            self.setQxInvPol()
#            self.CalcKernelMatrix(rms)
        self.CalcJacobianAntenna(Gains)
        Pa=P[self.iAnt]
        self.PrepareJHJ_EKF(Pa,rms)
        NPars=Pa.shape[0]
        PaOnes=np.diag(np.ones((NPars,),self.CType))

        evPa=np.zeros_like(Pa)

        for iPar in range(Pa.shape[0]):
            J_Px=self.J_x(PaOnes[iPar,:])
            xP=self.ApplyK_vec(J_Px,rms,Pa,DoReg=False)
            evPa[iPar,:]=xP.flatten()


        evPa= PaOnes-evPa#(np.diag(np.diag(Pa-Pa_new)))#Pa-Pa_new#np.abs(np.diag(np.diag(Pa-Pa_new)))
        evPa=np.diag(np.diag(evPa))
        #print evPa.min(),evPa.real.min()
        return evPa
           
            
    def doLMStep(self,Gains):
            
        
        T=ClassTimeIt.ClassTimeIt("doLMStep")
        T.disable()

#         A=np.random.randn(10000,100)+1j*np.random.randn(10000,100)
#         B=np.random.randn(10000,100)+1j*np.random.randn(10000,100)
#         AT=A.T#.conj().copy()
# #        AT=A.T
#         # A=np.require(A,requirements='F_CONTIGUOUS')
#         # AT=np.require(AT,requirements='F_CONTIGUOUS')
#         # A=np.require(A,requirements='F')
#         # AT=np.require(AT,requirements='F')


#         T=ClassTimeIt.ClassTimeIt("doLMStep")
#         for i in range(20):
#             np.dot(AT,B)

#         T.timeit("%i"%i)

        
        if not(self.HasKernelMatrix):
            self.CalcKernelMatrix()
            self.SelectChannelKernelMat()
            T.timeit("CalcKernelMatrix")

        Ga=self.GiveSubVecGainAnt(Gains)

        f=(self.DicoData["flags_flat"]==0)
        # ind=np.where(f)[0]
        # if self.iAnt==56:
        #     print ind.size/float(f.size),np.abs(Gains[self.iAnt,0,0,0])

        if self.DataAllFlagged:
            return Ga.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),None,{"std":-1.,"max":-1.,"kapa":None}



        # if ind.size==0:
        #     return Ga.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),None,{"std":-1.,"max":-1.,"kapa":None}


        z=self.DicoData["data_flat"]#self.GiveDataVec()
        self.CalcJacobianAntenna(Gains)
        T.timeit("CalcJacobianAntenna")
        self.PrepareJHJ_LM()
        T.timeit("PrepareJHJ_L")



        T.timeit("GiveSubVecGainAnt")
        Jx=self.J_x(Ga)
        T.timeit("Jx")
        zr=z-Jx
        zr[self.DicoData["flags_flat"]]=0
        T.timeit("resid")

        # JH_z_0=np.load("LM.npz")["JH_z"]
        # x1_0=np.load("LM.npz")["x1"]
        # z_0=np.load("LM.npz")["z"]
        # Jx_0=np.load("LM.npz")["Jx"]




        InfoNoise={"std":np.std(zr[f]),"max":np.max(np.abs(zr[f])),"kapa":None}


        JH_z=self.JH_z(zr)
        T.timeit("JH_z")
        #self.JHJinv=ModLinAlg.invSVD(self.JHJ)
        #self.JHJinv=np.linalg.inv(self.JHJ)
        xi=Ga.flatten()
        T.timeit("self.JHJinv_x")
        

        if self.DoTikhonov:
            self.LambdaTkNorm
            Gi=xi.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
            JH_z=JH_z.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
            for polIndex in range(self.NJacobBlocks_X):
                gireg=Gi[:,polIndex,:]
                #gi=JH_z[:,polIndex,:]
                x0reg=self.X0[:,polIndex,:]
                Linv=(self.Linv[:,polIndex,:])
                JH_z[:,polIndex,:]-=self.LambdaTkNorm*Linv*(gireg-x0reg)
        dx = (1./(1.+self.LambdaLM)) * self.JHJinv_x(JH_z)

        
        
        
        # if self.iAnt==5:
        #     f=(self.DicoData["flags_flat"]==0)
        #     pylab.figure(2)
        #     pylab.clf()
        #     pylab.plot((z[f])[::1])#[::11])
        #     pylab.plot((Jx[f])[::1])#[::11])
        #     pylab.plot(zr[f][::1])#[::11])
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     #stop

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

        dx+=xi
        del(self.LJacob)
        T.timeit("rest")
        # print self.iAnt,np.mean(x1),x1.size,ind.size

        return dx.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y)),None,InfoNoise

                                        
    def JHJinv_x(self,Gains):
        G=[]
        #nd,_,_=Gains.shape
        Gains=Gains.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        for polIndex in range(self.NJacobBlocks_X):
            Gain=Gains[:,polIndex,:]
            #print "JHJinv_x: %i %s . %s "%(polIndex,str(self.L_JHJinv[polIndex].shape),str(Gain.flatten().shape))
            Vec=np.dot(self.L_JHJinv[polIndex],Gain.flatten())
            Vec=Vec.reshape((self.NDir,1,self.NJacobBlocks_Y))
            G.append(Vec)
            
        Gout=np.concatenate(G,axis=1)
        #print "JHJinv_x: Gout %s "%(str(Gout.shape))
        
        return Gout.flatten()



    def Msq_x(self,LM,Gains):
        G=[]
        Gains=Gains.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        for polIndex in range(self.NJacobBlocks_X):
            Gain=Gains[:,polIndex,:]
            #print "Msq_x: %i %s . %s"%(polIndex,str(LM[polIndex].shape),str(Gain.flatten().shape))
            Vec=np.dot(LM[polIndex],Gain.flatten())
            Vec=Vec.reshape((self.NDir,1,self.NJacobBlocks_Y))
            G.append(Vec)
            
        Gout=np.concatenate(G,axis=1)
        #print "Msq_x: Gout %s "%(str(Gout.shape))
        
        return Gout.flatten()




    def JH_z(self,zin):
        #z=zin.reshape((self.NJacobBlocks,zin.size/self.NJacobBlocks))
        #z=zin.reshape((1,zin.size))
        Gains=np.zeros((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y),self.CType)
        for polIndex in range(self.NJacobBlocks_X):
            Jacob=self.LJacob[polIndex]
            
            flags=self.DicoData["flags_flat"][polIndex]
            ThisZ=zin[polIndex][flags==0]#self.DicoData["flags_flat"[polIndex]
            
            J=Jacob[flags==0]


            ThisZ=ThisZ.flatten()

            if self.TypeDot=="Numpy":
                Gain=np.dot(J.T.conj(),ThisZ)
            elif self.TypeDot=="SSE":
                ThisZ=ThisZ.reshape((1,ThisZ.size))
                JTc=self.LJacobTc[polIndex]#.copy()
                Gain=NpDotSSE.dot_A_BT(JTc,ThisZ)


            Gains[:,polIndex,:]=Gain.reshape((self.NDir,self.NJacobBlocks_Y))

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


    def J_x(self,Gains):
        z=[]
        Gains=Gains.reshape((self.NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        for polIndex in range(self.NJacobBlocks_X):
            Jacob=self.LJacob[polIndex]
            
            Gain=Gains[:,polIndex,:].flatten()

            #flags=self.DicoData["flags_flat"][polIndex]
            J=Jacob#[flags==0]
            # print J.shape, Gain.shape

            # # Numpy

            if self.TypeDot=="Numpy":
                Z=np.dot(J,Gain)
            elif self.TypeDot=="SSE":
                Gain=Gain.reshape((1,Gain.size))
                Z=NpDotSSE.dot_A_BT(J,Gain).ravel()

            z.append(Z)



        z=np.array(z)
        return z

    def PredictOrigFormat(self,GainsIn):
        if self.GD["VisData"]["FreePredictGainColName"]!=None:
            self.PredictOrigFormat_Type(GainsIn,Type="Gains")
        if self.GD["VisData"]["FreePredictColName"]!=None:
            self.PredictOrigFormat_Type(GainsIn,Type="NoGains")


    def PredictOrigFormat_Type(self,GainsIn,Type="Gains"):
        #print "    COMPUTE PredictOrigFormat"
        Gains=GainsIn.copy()
        na,nd,_,_=Gains.shape
        #Type="NoGains"
        if Type=="NoGains":
            if self.PolMode=="Scalar":
                Gains=np.ones((na,nd,1,1),np.complex64)
            elif self.PolMode=="IDiag":
                Gains=np.ones((na,nd,2,1),np.complex64)
            else:
                Gains=np.zeros((na,nd,2,2),np.complex64)
                Gains[:,0,0]=1
                Gains[:,1,1]=1
            NameShmData="%sPredictedData"%self.IdSharedMem
            NameShmIndices="%sIndicesData"%self.IdSharedMem
        elif Type=="Gains":
            NameShmData="%sPredictedDataGains"%self.IdSharedMem
            NameShmIndices="%sIndicesDataGains"%self.IdSharedMem
        
            
        PredictedData=NpShared.GiveArray(NameShmData)
        Indices=NpShared.GiveArray(NameShmIndices)

        Ga=self.GiveSubVecGainAnt(Gains).copy()

        self.CalcJacobianAntenna(Gains)
        self.PrepareJHJ_LM()
        zp=self.J_x(Ga)#self.DicoData["data_flat"]#
        DicoData=self.DicoData

        nr,nch,_,_=DicoData["flags"].shape
            
        indRowsThisChunk=self.DATA["indRowsThisChunk"]
        indOrig=DicoData["indOrig"]
        indThis=np.arange(DicoData["indOrig"].size)

        IndicesSel0=Indices[indRowsThisChunk,:,:][indOrig,self.ch0:self.ch1,0].ravel()
        IndicesSel1=Indices[indRowsThisChunk,:,:][indOrig,self.ch0:self.ch1,1].ravel()
        IndicesSel2=Indices[indRowsThisChunk,:,:][indOrig,self.ch0:self.ch1,2].ravel()
        IndicesSel3=Indices[indRowsThisChunk,:,:][indOrig,self.ch0:self.ch1,3].ravel()
        
        D=np.rollaxis(zp.reshape(self.NJacobBlocks_X,nr,nch,self.NJacobBlocks_Y),0,3).reshape(nr,nch,self.NJacobBlocks_X,self.NJacobBlocks_Y)

        if self.PolMode=="Scalar":
            # PredictedData.ravel()[IndicesSel0]=D[indThis,:,0,0].ravel()
            PredictedData.flat[IndicesSel0]=D[indThis,:,0,0].ravel()
            PredictedData.flat[IndicesSel3]=D[indThis,:,0,0].ravel()
        elif self.PolMode=="IDiag":
            PredictedData.flat[IndicesSel0]=D[indThis,:,0,0].ravel()
            PredictedData.flat[IndicesSel3]=D[indThis,:,1,0].ravel()
        elif self.PolMode=="IFull":
            PredictedData.flat[IndicesSel0]=D[indThis,:,0,0].ravel()
            PredictedData.flat[IndicesSel1]=D[indThis,:,0,1].ravel()
            PredictedData.flat[IndicesSel2]=D[indThis,:,1,0].ravel()
            PredictedData.flat[IndicesSel3]=D[indThis,:,1,1].ravel()


        # d0=self.DATA["data"]#[indOrig,:,0]
        # #d1=D[indThis,:,0,0]
        # d2=PredictedData[indRowsThisChunk,:,:]#[indOrig,:,0]

        # pylab.clf()
        # pylab.plot(d0[:,2,0].real)
        # # pylab.plot(d1[:,2,0].real)
        # pylab.plot(d2[:,2,0].real)
        # pylab.plot((d0-d2)[:,2,0].real)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # # stop




    def CalcJacobianAntenna(self,GainsIn):
        if not(self.HasKernelMatrix): stop
        iAnt=self.iAnt
        NDir=self.NDir
        n4vis=self.n4vis
        #print "n4vis",n4vis
        na=self.na
        #print GainsIn.shape,na,NDir,self.NJacobBlocks,self.NJacobBlocks
        Gains=GainsIn.reshape((na,NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y))
        Jacob=np.zeros((n4vis,self.NJacobBlocks_Y,NDir,self.NJacobBlocks_Y),self.CType)

        if (self.PolMode=="IFull")|(self.PolMode=="Scalar"):
            self.LJacob=[Jacob]*self.NJacobBlocks_X
        elif self.PolMode=="IDiag":
            self.LJacob=[Jacob,Jacob.copy()]
        LJacob=self.LJacob
        
        for iDir in range(NDir):
            G=Gains[self.A1,iDir].conj()

            K_XX=self.K_XX[iDir]
            K_YY=self.K_YY[iDir]

            nr=G.shape[0]

            if self.PolMode=="Scalar":
                J0=Jacob[:,0,iDir,0]
                g0_conj=G[:,0,0].reshape((nr,1))
                J0[:]=(g0_conj*K_XX).reshape((K_XX.size,))

            
            elif self.PolMode=="IFull":
                J0=Jacob[:,0,iDir,0]
                g0_conj=G[:,0,0].reshape((nr,1))
                J0[:]=(g0_conj*K_XX).reshape((K_XX.size,))

                J1=Jacob[:,0,iDir,1]
                J2=Jacob[:,1,iDir,0]
                J3=Jacob[:,1,iDir,1]
                g1_conj=G[:,1,0].reshape((nr,1))
                g2_conj=G[:,0,1].reshape((nr,1))
                g3_conj=G[:,1,1].reshape((nr,1))

                J1[:]=(g2_conj*K_YY).reshape((K_XX.size,))
                J2[:]=(g1_conj*K_XX).reshape((K_XX.size,))
                J3[:]=(g3_conj*K_YY).reshape((K_XX.size,))

            elif self.PolMode=="IDiag":
                J0=LJacob[0][:,0,iDir,0]
                g0_conj=G[:,0,0].reshape((nr,1))
                J0[:]=(g0_conj*K_XX).reshape((K_XX.size,))

                J1=LJacob[1][:,0,iDir,0]
                g1_conj=G[:,1,0].reshape((nr,1))
                J1[:]=(g1_conj*K_YY).reshape((K_XX.size,))


        for J in LJacob:
            J.shape=(n4vis*self.NJacobBlocks_Y,NDir*self.NJacobBlocks_Y)


        self.LJacobTc=[]
        for polIndex in range(self.NJacobBlocks_X):
            flags=self.DicoData["flags_flat"][polIndex]
            J=self.LJacob[polIndex][flags==0]
            self.LJacobTc.append(J.T.conj().copy())

        self.L_JHJ=[]
        for polIndex in range(self.NJacobBlocks_X):
            flags=self.DicoData["flags_flat"][polIndex]
            J=self.LJacob[polIndex][flags==0]
            nrow,_=J.shape
            self.nrow_nonflagged=nrow
            JH=J.T.conj()
            if type(self.Rinv_flat)!=type(None):
                Rinv=self.Rinv_flat[polIndex][flags==0].reshape((nrow,1))

                if self.TypeDot=="Numpy":
                    JHJ=np.dot(J.T.conj(),Rinv*J)
                elif self.TypeDot=="SSE":
                    RinvJ_T=(Rinv*J).T.copy()
                    JTc=self.LJacobTc[polIndex]#.copy()
                    JHJ=NpDotSSE.dot_A_BT(JTc,RinvJ_T)

            else:
                if self.TypeDot=="Numpy":
                    JHJ=np.dot(J.T.conj(),J)
                elif self.TypeDot=="SSE":
                    J_T=J.T.copy()
                    JTc=self.LJacobTc[polIndex]#.copy()
                    JHJ=NpDotSSE.dot_A_BT(JTc,J_T)
                

            self.L_JHJ.append(self.CType(JHJ))


        # self.JHJinv=np.linalg.inv(self.JHJ)
        # self.JHJinv=np.diag(np.diag(self.JHJinv))

    def CalcKernelMatrix(self,rms=0.):
        # Out[28]: ['freqs', 'times', 'A1', 'A0', 'flags', 'uvw', 'data']
        T=ClassTimeIt.ClassTimeIt("CalcKernelMatrix Ant=%i"%self.iAnt)
        T.disable()
        DATA=self.DATA
        iAnt=self.iAnt
        na=int(DATA['infos'][0])
        self.na=na
        NDir=self.SM.NDir
        self.NDir=NDir
        self.iAnt=iAnt



        T.timeit("stuff")
        
        self.DicoData=self.GiveData(DATA,iAnt,rms=rms)

        T.timeit("data")
        # self.Data=self.DicoData["data"]
        self.A1=self.DicoData["A1"]
        # print "AntMax1",self.SharedDataDicoName,np.max(self.A1)
        # print self.DicoData["A1"]
        # print "AntMax0",self.SharedDataDicoName,np.max(self.DicoData["A0"])
        # print self.DicoData["A0"]
        nrows,nchan,_,_=self.DicoData["flags"].shape
        n4vis=nrows*nchan
        self.n4vis=n4vis
        
        KernelSharedName="%sKernelMat.%2.2i"%(self.IdSharedMem,self.iAnt)
        self.KernelMat_AllChan=NpShared.GiveArray(KernelSharedName)

        if type(self.KernelMat_AllChan)!=type(None):
            self.HasKernelMatrix=True
            if self.PolMode=="IFull":
                self.K_XX_AllChan=self.KernelMat_AllChan[0]
                self.K_YY_AllChan=self.KernelMat_AllChan[1]
                self.NJacobBlocks_X=2
                self.NJacobBlocks_Y=2
            elif self.PolMode=="Scalar":
                #n4vis=self.DicoData["data_flat"].size
                self.K_XX_AllChan=self.KernelMat_AllChan[0]
                self.K_YY_AllChan=self.K_XX_AllChan
                #self.n4vis=n4vis
                self.NJacobBlocks_X=1
                self.NJacobBlocks_Y=1
            elif self.PolMode=="IDiag":
                #n4vis=self.DicoData["data_flat"].size
                self.K_XX_AllChan=self.KernelMat_AllChan[0]
                self.K_YY_AllChan=self.KernelMat_AllChan[1]
                #self.n4vis=n4vis
                self.NJacobBlocks_X=2
                self.NJacobBlocks_Y=1
            # self.Data=self.Data.reshape((nrows,nchan,self.NJacobBlocks,self.NJacobBlocks))

            #print "Kernel From shared"
            return
        else:
            #print "    COMPUTE KERNEL"
            pass

        T.timeit("stuff 2")
        # GiveArray(Name)
        nchan_AllChan=self.DicoData["freqs_full"].size
        n4vis_AllChan=nrows*nchan_AllChan
        self.n4vis_AllChan=n4vis_AllChan
            
        if self.PolMode=="IFull":
            #self.K_XX=np.zeros((NDir,n4vis/nchan,nchan),np.complex64)
            #self.K_YY=np.zeros((NDir,n4vis/nchan,nchan),np.complex64)
            self.KernelMat_AllChan=NpShared.zeros(KernelSharedName,(2,NDir,n4vis_AllChan/nchan_AllChan,nchan_AllChan),dtype=self.CType)
            self.K_XX_AllChan=self.KernelMat_AllChan[0]
            self.K_YY_AllChan=self.KernelMat_AllChan[1]
            # KernelMatrix=NpShared.zeros(KernelSharedName,(n4vis,NDir,2),dtype=np.complex64)
            self.NJacobBlocks_X=2
            self.NJacobBlocks_Y=2
        elif self.PolMode=="Scalar":
            #n4vis=self.Data.size
            # KernelMatrix_XX=np.zeros((NDir,n4vis,nchan),np.complex64)
            # KernelMatrix=NpShared.zeros(KernelSharedName,(n4vis,NDir,1),dtype=np.complex64)
            self.KernelMat_AllChan=NpShared.zeros(KernelSharedName,(1,NDir,n4vis_AllChan/nchan_AllChan,nchan_AllChan),dtype=self.CType)
            self.K_XX_AllChan=self.KernelMat_AllChan[0]
            self.K_YY_AllChan=self.K_XX_AllChan
            self.NJacobBlocks_X=1
            self.NJacobBlocks_Y=1
        elif self.PolMode=="IDiag":
            self.KernelMat_AllChan=NpShared.zeros(KernelSharedName,(2,NDir,n4vis_AllChan/nchan_AllChan,nchan_AllChan),dtype=self.CType)
            self.K_XX_AllChan=self.KernelMat_AllChan[0]
            self.K_YY_AllChan=self.KernelMat_AllChan[1]
            self.NJacobBlocks_X=2
            self.NJacobBlocks_Y=1
        T.timeit("stuff 3")
            
        #self.Data=self.Data.reshape((nrows,nchan,self.NJacobBlocks,self.NJacobBlocks))

        #self.K_XX=[]
        #self.K_YY=[]

        ApplyTimeJones=None
        #print self.DicoData.keys()
        if "DicoPreApplyJones" in self.DicoData.keys():
            ApplyTimeJones=self.DicoData["DicoPreApplyJones"]

        #import gc
        #gc.enable()
        # gc.set_debug(gc.DEBUG_LEAK)


        # ##############################################
        # from SkyModel.Sky import ClassSM
        # SM=ClassSM.ClassSM("ModelRandom00.4.txt.npy")
        # SM.Type="Catalog"
        # SM.Calc_LM(self.SM.rac,self.SM.decc)
        # self.KernelMat1=np.zeros((1,NDir,n4vis/nchan,nchan),dtype=self.CType)
        # self.K1_XX=self.KernelMat1[0]
        # self.K1_YY=self.K1_XX
        # import pylab
        # pylab.figure(0)
        # pylab.clf()
        # pylab.figure(1)
        # pylab.clf()
        # pylab.figure(0)
        
        


        for iDir in range(NDir):
            

            K=self.PM.predictKernelPolCluster(self.DicoData,self.SM,iDirection=iDir,ApplyTimeJones=ApplyTimeJones)
            #K=self.PM.predictKernelPolCluster(self.DicoData,self.SM,iDirection=iDir)#,ApplyTimeJones=ApplyTimeJones)
            #K*=-1
            T.timeit("Calc K0")


                #gc.collect()
                #print gc.garbage


            # if (iDir==31)&(self.iAnt==51):
            #     ifile=0
            #     while True:
            #         fname="png/Kernel.%5.5i.npy"%ifile
            #         if not(os.path.isfile(fname)) :
            #             np.save(fname,K)
            #             break
            #         ifile+=1

            K_XX=K[:,:,0]
            K_YY=K[:,:,3]
            if self.PolMode=="Scalar":
                K_XX=(K_XX+K_YY)/2.
                K_YY=K_XX

            self.K_XX_AllChan[iDir,:,:]=K_XX
            self.K_YY_AllChan[iDir,:,:]=K_YY
            #self.K_XX.append(K_XX)
            #self.K_YY.append(K_YY)



        #     ######################
        #     K1=self.PM.predictKernelPolCluster(self.DicoData,SM,iDirection=iDir)#,ApplyTimeJones=ApplyTimeJones)

        #     A0=self.DicoData["A0"]
        #     A1=self.DicoData["A1"]
        #     ind=np.where((A0==0)&(A1==26))[0]
        #     d1=K[ind,0,0] 
        #     d0=K1[ind,0,0]
        #     #op0=np.abs
        #     op0=np.real
        #     #op1=np.imag
        #     pylab.figure(0)
        #     pylab.subplot(1,NDir,iDir+1)
        #     pylab.plot(op0(d0))
        #     pylab.plot(op0(d1))
        #     #pylab.plot(op0(d1)/op0(d0))
        #     pylab.ylim(-15,15)
        #     pylab.draw()
        #     pylab.show(False)

        #     op1=np.angle
        #     pylab.figure(1)
        #     pylab.subplot(1,NDir,iDir+1)
        #     pylab.plot(op1(d0))
        #     pylab.plot(op1(d1))
        #     #pylab.plot(op1(d1)-op1(d0))
        #     pylab.ylim(-np.pi,np.pi)

        #     # pylab.subplot(2,1,2)
        #     # #pylab.plot(op1(d0))
        #     # pylab.plot(op1(d1*d0.conj()))#,ls="--")
        #     # #pylab.plot(op1(d0*d1.conj()),ls="--")
        #     # #pylab.ylim(-1,1)
        #     pylab.draw()
        #     pylab.show(False)
            
        # #     K1_XX=K1[:,:,0]
        # #     K1_YY=K1[:,:,3]
        # #     if self.PolMode=="Scalar":
        # #         K1_XX=(K1_XX+K1_YY)/2.
        # #         K1_YY=K1_XX

        # #     self.K1_XX[iDir,:,:]=K1_XX
        # #     self.K1_YY[iDir,:,:]=K1_YY
        # #     #self.K_XX.append(K_XX)
        # #     #self.K_YY.append(K_YY)

        # #     del(K1,K1_XX,K1_YY)
        # #     del(K,K_XX,K_YY)



        # 
        #stop
        #gc.collect()
        self.HasKernelMatrix=True
        T.timeit("stuff 4")

    def SelectChannelKernelMat(self):
        self.K_XX=self.K_XX_AllChan[:,:,self.ch0:self.ch1]
        self.K_YY=self.K_YY_AllChan[:,:,self.ch0:self.ch1]

        


        NDir=self.SM.NDir
        for iDir in range(NDir):
            
            K=self.K_XX[iDir,:,:]

            indRow,indChan=np.where(K==0)
            self.DicoData["flags"][indRow,indChan,:]=1
        DicoData=self.DicoData
        nr,nch=K.shape
        flags_flat=np.rollaxis(DicoData["flags"],2).reshape(self.NJacobBlocks_X,nr*nch*self.NJacobBlocks_Y)
        DicoData["flags_flat"][flags_flat]=1


        self.DataAllFlagged=False
        NP,_=DicoData["flags_flat"].shape
        for ipol in range(NP):
            f=(DicoData["flags_flat"][ipol]==0)
            ind=np.where(f)[0]
            if ind.size==0: 
                self.DataAllFlagged=True
                continue
            fracFlagged=ind.size/float(f.size)
            if fracFlagged<0.2:#ind.size==0:
                self.DataAllFlagged=True


        #print "SelectChannelKernelMat",np.count_nonzero(DicoData["flags_flat"]),np.count_nonzero(DicoData["flags"])



    def GiveData(self,DATA,iAnt,rms=0.):
        
        #DicoData=NpShared.SharedToDico(self.SharedDataDicoName)
        if self.SharedDicoDescriptors["SharedAntennaVis"]==None:
            #print "     COMPUTE DATA"
            DicoData={}
            ind0=np.where(DATA['A0']==iAnt)[0]
            ind1=np.where(DATA['A1']==iAnt)[0]
            DicoData["A0"] = np.concatenate([DATA['A0'][ind0], DATA['A1'][ind1]])
            DicoData["A1"] = np.concatenate([DATA['A1'][ind0], DATA['A0'][ind1]])
            D0=DATA['data'][ind0,self.ch0:self.ch1]
            D1=DATA['data'][ind1,self.ch0:self.ch1].conj()
            c1=D1[:,:,1].copy()
            c2=D1[:,:,2].copy()
            D1[:,:,1]=c2
            D1[:,:,2]=c1
            DicoData["data"] = np.concatenate([D0, D1])
            DicoData["indOrig"] = ind0
            DicoData["indOrig1"] = ind1
            DicoData["uvw"]  = np.concatenate([DATA['uvw'][ind0], -DATA['uvw'][ind1]])
            DicoData["UVW_dt"]  = np.concatenate([DATA["UVW_dt"][ind0], -DATA["UVW_dt"][ind1]])

            if "W" in DATA.keys():
                DicoData["W"] = np.concatenate([DATA['W'][ind0,self.ch0:self.ch1], DATA['W'][ind1,self.ch0:self.ch1]])

            # DicoData["IndexTimesThisChunk"]=np.concatenate([DATA["IndexTimesThisChunk"][ind0], DATA["IndexTimesThisChunk"][ind1]]) 
            # DicoData["UVW_RefAnt"]=DATA["UVW_RefAnt"][it0:it1]

            if "Kp" in DATA.keys():
                 DicoData["Kp"]=DATA["Kp"]

            D0=DATA['flags'][ind0,self.ch0:self.ch1]
            D1=DATA['flags'][ind1,self.ch0:self.ch1].conj()
            c1=D1[:,:,1].copy()
            c2=D1[:,:,2].copy()
            D1[:,:,1]=c2
            D1[:,:,2]=c1
            DicoData["flags"] = np.concatenate([D0, D1])



            if self.SM.Type=="Image":
                #DicoData["flags_image"]=DicoData["flags"].copy()
                nr,_,_=DicoData["data"].shape
                _,nch,_=DATA['data'].shape
                DicoData["flags_image"]=np.zeros((nr,nch,4),np.bool8)
                #DicoData["flags_image"].fill(0)

            nr,nch,_=DicoData["data"].shape
            
            if self.PolMode=="Scalar":
                d=(DicoData["data"][:,:,0]+DicoData["data"][:,:,-1])/2
                DicoData["data"] = d.reshape((nr,nch,1))
                f=(DicoData["flags"][:,:,0]|DicoData["flags"][:,:,-1])
                DicoData["flags"] = f.reshape((nr,nch,1))
            elif self.PolMode=="IDiag":
                d=DicoData["data"][:,:,0::3]
                DicoData["data"] = d.copy().reshape((nr,nch,2))
                f=DicoData["flags"][:,:,0::3]
                DicoData["flags"] = f.copy().reshape((nr,nch,2))



            DicoData["freqs"]   = DATA['freqs'][self.ch0:self.ch1]
            DicoData["dfreqs"]   = DATA['dfreqs'][self.ch0:self.ch1]
            DicoData["times"] = np.concatenate([DATA['times'][ind0], DATA['times'][ind1]])
            DicoData["infos"] = DATA['infos']

            # nr,nch,_=DicoData["data"].shape

            FlagsShape=DicoData["flags"].shape
            FlagsSize=DicoData["flags"].size
            DicoData["flags"]=DicoData["flags"].reshape(nr,nch,self.NJacobBlocks_X,self.NJacobBlocks_Y)
            DicoData["data"]=DicoData["data"].reshape(nr,nch,self.NJacobBlocks_X,self.NJacobBlocks_Y)

            DicoData["flags_flat"]=np.rollaxis(DicoData["flags"],2).reshape(self.NJacobBlocks_X,nr*nch*self.NJacobBlocks_Y)
            DicoData["data_flat"]=np.rollaxis(DicoData["data"],2).reshape(self.NJacobBlocks_X,nr*nch*self.NJacobBlocks_Y)



            # ###################
            # NJacobBlocks_X=2
            # NJacobBlocks_Y=2
            # F0=np.zeros((nr,nch,NJacobBlocks_X,NJacobBlocks_Y))
            # FlagsShape=F0.shape
            # FlagsSize=F0.size
            # F0=np.arange(FlagsSize).reshape(FlagsShape)
            # F0Flat=np.rollaxis(F0,2).reshape(NJacobBlocks_X,nr*nch*NJacobBlocks_Y)
            # F1=np.rollaxis(F0Flat.reshape(NJacobBlocks_X,nr,nch,NJacobBlocks_Y),0,3).reshape(FlagsShape)
            # print np.count_nonzero((F0-F1).ravel())
            # stop
            # ###################


            del(DicoData["data"])


            if rms!=0.:
                DicoData["rms"]=np.array([rms],np.float32)
                u,v,w=DicoData["uvw"].T
                if self.ResolutionRad!=None:
                    freqs=DicoData["freqs"]
                    wave=np.mean(299792456./freqs)
                    d=np.sqrt((u/wave)**2+(v/wave)**2)
                    FWHMFact=2.*np.sqrt(2.*np.log(2.))
                    sig=self.ResolutionRad/FWHMFact
                    V=(1./np.exp(-d**2*np.pi*sig**2))**2
                    
                    V=V.reshape((V.size,1,1))*np.ones((1,freqs.size,self.npolData))
                else:
                    V=np.ones((u.size,freqs.size,self.npolData),np.float32)
                    
                if "W" in DicoData.keys():
                    W=DicoData["W"]**2
                    W_nrows,W_nch=W.shape
                    W[W==0]=1.e-6
                    V=V/W.reshape((W_nrows,W_nch,1))
                    
                R=rms**2*V
                
                Rinv=1./R
                
                self.R_flat=np.rollaxis(R,2).reshape(self.NJacobBlocks_X,nr*nch*self.NJacobBlocks_Y)
                self.Rinv_flat=np.rollaxis(Rinv,2).reshape(self.NJacobBlocks_X,nr*nch*self.NJacobBlocks_Y)

                self.R_flat=np.require(self.R_flat,dtype=self.CType)
                self.Rinv_flat=np.require(self.Rinv_flat,dtype=self.CType)

                Rmin=np.min(R)
                #Rmax=np.max(R)
                Flag=(self.R_flat>1e3*Rmin)
                DicoData["flags_flat"][Flag]=1
                DicoData["Rinv_flat"]=self.Rinv_flat
                DicoData["R_flat"]=self.R_flat

            self.DataAllFlagged=False
            NP,_=DicoData["flags_flat"].shape
            for ipol in range(NP):
                f=(DicoData["flags_flat"][ipol]==0)
                ind=np.where(f)[0]
                
                if ind.size==0: 
                    self.DataAllFlagged=True
                    continue

                fracFlagged=ind.size/float(f.size)
                if fracFlagged<0.2:#ind.size==0:
                    self.DataAllFlagged=True

            DicoData=NpShared.DicoToShared(self.SharedDataDicoName,DicoData)
            self.SharedDicoDescriptors["SharedAntennaVis"]=NpShared.SharedDicoDescriptor(self.SharedDataDicoName,DicoData)
        else:
            DicoData=NpShared.SharedObjectToDico(self.SharedDicoDescriptors["SharedAntennaVis"])
            if rms!=0.:
                self.Rinv_flat=DicoData["Rinv_flat"]
                self.R_flat=DicoData["R_flat"]

            #print "DATA From shared"
            #print np.max(DicoData["A0"])
            #np.save("testA0",DicoData["A0"])
            #DicoData["A0"]=np.load("testA0.npy")
            #DicoData=NpShared.SharedToDico(self.SharedDataDicoName)
            #print np.max(DicoData["A0"])
            #print

            #stop

        if "DicoPreApplyJones" in DATA.keys():
            DicoJonesMatrices={}
            ind0=DicoData["indOrig"]
            ind1=DicoData["indOrig1"]
            #DicoApplyJones=NpShared.SharedToDico("%sPreApplyJonesFile"%self.IdSharedMem)
            
            DicoJonesMatrices["DicoApplyJones"]=DATA["DicoPreApplyJones"]
            DicoJonesMatrices["DicoApplyJones"]["DicoClusterDirs"]=DATA["DicoClusterDirs"]
            MapTimes=DATA["Map_VisToJones_Time"]
            MapTimesSel=np.concatenate([MapTimes[ind0], MapTimes[ind1]])
            DicoJonesMatrices["DicoApplyJones"]["Map_VisToJones_Time"]=MapTimesSel


            DicoData["DicoPreApplyJones"]=DicoJonesMatrices
            #print DATA["Map_VisToJones_Time"].max()
            #stop

        self.DoTikhonov=False
        #self.GD["CohJones"]["LambdaTk"]=0
        if (self.GD["CohJones"]["LambdaTk"]!=0)&(self.GD["Solvers"]["SolverType"]=="CohJones"):
            self.DoTikhonov=True
            self.LambdaTk=self.GD["CohJones"]["LambdaTk"]
            self.Linv=NpShared.GiveArray("%sLinv"%self.IdSharedMem)
            self.X0=NpShared.GiveArray("%sX0"%self.IdSharedMem)
            

        # DicoData["A0"] = np.concatenate([DATA['A0'][ind0]])
        # DicoData["A1"] = np.concatenate([DATA['A1'][ind0]])
        # D0=DATA['data'][ind0]
        # DicoData["data"] = np.concatenate([D0])
        # DicoData["uvw"]  = np.concatenate([DATA['uvw'][ind0]])
        # DicoData["flags"] = np.concatenate([DATA['flags'][ind0]])
        # DicoData["freqs"]   = DATA['freqs']

        self.DataAllFlagged=False
        NP,_=DicoData["flags_flat"].shape
        for ipol in range(NP):
            f=(DicoData["flags_flat"][ipol]==0)
            ind=np.where(f)[0]
            if ind.size==0: 
                self.DataAllFlagged=True
                continue
            fracFlagged=ind.size/float(f.size)
            if fracFlagged<0.2:#ind.size==0:
                self.DataAllFlagged=True

        DicoData["freqs_full"]   = self.DATA['freqs']
        DicoData["dfreqs_full"]   = self.DATA['dfreqs']

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
    PolMode="IFull"#"Scalar"
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
    
