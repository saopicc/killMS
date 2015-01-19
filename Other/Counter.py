import numpy as np

class Counter():
    def __init__(self,N=1):
        if N==0: N=1
        self.N=N
        self.i=-1

    def __call__(self):
        i=self.i
        #print "%i -> %i"%(i,i+1)
        self.i+=1
        return (self.i%self.N==0)


class CounterTime():
    def __init__(self,dt=60):
        self.dt=dt
        self.CurrentTime=-1e6

    def __call__(self,time):
        rep=False
        #print time,self.CurrentTime,(time-self.CurrentTime)
        if (np.abs(time-self.CurrentTime)>self.dt):
            self.CurrentTime=time+self.dt
            rep=True
        return rep

