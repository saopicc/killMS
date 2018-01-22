import numpy as np

def TECToPhase(TEC,freq):
    K=8.4479745e9
    phase=K*TEC*(1./freq)
    return phase

def TECToZ(TEC,ConstPhase,freq):
    return np.exp(1j*(TECToPhase(TEC,freq)+ConstPhase))

class ClassFitTEC():
    def __init__(self,gains,nu):
        self.nf,self.na=gains.shape
        self.G=gains
        self.nu=nu
        na=self.na
        self.nbl=(na**2-na)/2

        self.Y=(self.G.reshape((-1,1))*self.G.conj().reshape((1,-1))).ravel()
        A0,A1=np.mgrid[0:na:1,0:na:1]
        self.A0,self.A1=A0.ravel(),A1.ravel()
        
        self.nu_g=self.nu.reshape((-1,1))*np.ones((1,self.na))
        self.nu_Y=(self.nu_g.reshape((-1,1))*np.ones((1,self.nu_g.size))).ravel()
        
        
    def giveJacobian(self):
        A=np.range(self.na)
        J=np.zeros((self.Y.size,self.na*2),np.complex64)
        Jc=J[:,0:self.na]
        Jt=J[:,self.na:]
        
        
        # TEC Jacobian
        
