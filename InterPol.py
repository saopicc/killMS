import numpy as np
import pylab


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
    def __init__(self,FileName,Type="linear",Interval=6.,PolMode="Full"):
        self.FileName=FileName
        self.DicoFile=np.load(FileName)
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
        nt,na,nd,_,_=self.Sols.G.shape
        for iDir in range(nd):
            G=self.Sols.G[:,:,iDir,:,:]
            self.Sols.G[:,:,iDir,:,:]=NormMatrices(G)

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

        for iAnt in range(na):
            for iDir in range(nd):
                # Amplitude
                for ipol in Pols:
                    xp=Sols0.tm
                    yp=np.abs(G0[:,iAnt,iDir,ipol])
                    x=times
                    y=np.interp(x, xp, yp)
                    
                    pylab.clf()
                    pylab.scatter(xp,yp)
                    pylab.plot(x,y)
                    pylab.draw()
                    pylab.show(False)
                    pylab.pause(0.1)
                    


        
def test():
    FileName="Simul.npz"
    CI=ClassInterpol(FileName)
    CI.InterPol()
