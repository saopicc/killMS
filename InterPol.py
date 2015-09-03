import numpy as np
import pylab
import os
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassInterpol")

def NormMatrices(G):
    nt,na,_,_=G.shape
    for it in range(nt):
        Gt=G[it,:,:,:]
        u,s,v=np.linalg.svd(Gt[0])
        U=np.dot(u,v)
        for iAnt in range(0,na):
            Gt[iAnt,:,:]=np.dot(U.T.conj(),Gt[iAnt,:,:])
    return G

class ClassInterpol():
    def __init__(self,FileName,Type="linear",Interval=6.,PolMode="Full",OutName=None):
        self.FileName=os.path.abspath(FileName)
        self.OutName=OutName
        self.DicoFile=dict(np.load(FileName))
        self.Sols=self.DicoFile["Sols"]
        self.Sols=self.Sols.view(np.recarray)
        self.Interval=Interval
        self.PolMode=PolMode

        self.NormAllDirs()

        #self.StationNames=self.DicoFile["StationNames"]
        #self.SkyModel=self.DicoFile["SkyModel"]
        #self.ClusterCat=self.DicoFile["ClusterCat"]
        #self.SourceCatSub=self.DicoFile["SourceCatSub"]
        #self.ModelName=self.DicoFile["ModelName"]
    
    def NormAllDirs(self):
        print>>log,"Normalising Jones matrices ...."
        nt,na,nd,_,_=self.Sols.G.shape
        for iDir in range(nd):
            G=self.Sols.G[:,:,iDir,:,:]
            self.Sols.G[:,:,iDir,:,:]=NormMatrices(G)
        print>>log,"  Done"

    def InterPol(self):
        Sols0=self.Sols
        nt0,na,nd,_,_=Sols0.G.shape
        G0=Sols0.G.reshape((nt0,na,nd,4))

        T0=Sols0.t0[0]
        T1=Sols0.t1[-1]
        times=np.arange(T0,T1,self.Interval).tolist()
        if times[-1]<T1: times+=[T1]
        times=np.array(times)
        nt1=times.size-1

        Sols1=np.zeros(nt1,dtype=Sols0.dtype)
        Sols1=Sols1.view(np.recarray)
        nt1,na,nd,_,_=Sols1.G.shape
        G1=Sols1.G.reshape((nt1,na,nd,4))

        if self.PolMode=="Full":
            Pols=range(4)

        xp=Sols0.tm
        x=(times[0:-1]+times[1::])/2
        Sols1.t0=times[0:-1]
        Sols1.t1=times[1::]
        Sols1.tm=x
        
        for iDir in range(nd):
            print>>log,"Interpolating direction %3.3i in linear mode"%iDir
            for iAnt in range(na):
                for ipol in Pols:
                    # Amplitude
                    yp=np.abs(G0[:,iAnt,iDir,ipol])
                    y=np.interp(x, xp, yp)
                    G1[:,iAnt,iDir,ipol]=y[:]

                    # Phase
                    yp=np.angle(G0[:,iAnt,iDir,ipol])
                    y=np.interp(x, xp, yp)
                    G1[:,iAnt,iDir,ipol]*=np.exp(1j*y[:])

                # op0=np.abs
                # op1=np.angle
                # pylab.clf()
                # for ipol in Pols:
                #     pylab.subplot(2,1,1)
                #     pylab.scatter(xp,op0(G0[:,iAnt,iDir,ipol]))
                #     pylab.plot(x,op0(G1[:,iAnt,iDir,ipol]),marker=".",ls="")
                #     pylab.subplot(2,1,2)
                #     pylab.scatter(xp,op1(G0[:,iAnt,iDir,ipol]))
                #     pylab.plot(x,op1(G1[:,iAnt,iDir,ipol]),marker=".",ls="")

                # pylab.draw()
                # pylab.show(False)
                # pylab.pause(0.1)

        G1=G1.reshape((nt1,na,nd,2,2))
        Sols1.G=G1
        self.Sols1=Sols1
     
    def Save(self):
        self.DicoFile["Sols"]=self.Sols1
        OutName=self.OutName
        if OutName==None:
            FileName=self.FileName.split("/")[-1]
            Path="/".join(self.FileName.split("/")[0:-1])+"/"
            _,SolveType,_,_=self.FileName.split(".")
            OutName="%skillMS.%s.Interpol.sols.npz"%(Path,SolveType)
        print>>log,"Saving interpolated solutions in: %s"%OutName
        np.savez(OutName,**self.DicoFile)

        
def test():
    FileName="killMS.KAFCA.sols.npz"
    CI=ClassInterpol(FileName)
    CI.InterPol()
    CI.Save()
