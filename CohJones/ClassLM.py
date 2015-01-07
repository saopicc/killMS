
import ClassJacobian

def test():
    VS=ClassVisServer.ClassVisServer("../TEST/0000.MS/")

    MS=VS.MS
    SM=ClassSM.ClassSM("../TEST/ModelRandom00.txt.npy")
    SM.Calc_LM(MS.rac,MS.decc)




    nd=SM.NDir
    npol=4
    na=MS.na
    Gains=np.zeros((na,nd,npol),dtype=np.complex64)
    Gains[:,:,0]=1j
    Gains[:,:,-1]=1
    Gains+=np.random.randn(*Gains.shape)*0.5+1j*np.random.randn(*Gains.shape)
    Gains=np.random.randn(*Gains.shape)+1j*np.random.randn(*Gains.shape)
    DATA=VS.GiveNextVis(0,50)

    # Apply Jones
    PM=ClassPredict(Precision="S")
    DATA["data"]=PM.predictKernelPolCluster(DATA,SM,ApplyJones=Gains)
    
    ############################

    y=JM.GiveDataVec()
    
#    Gain=JM.ThisGain[:,1,:]
    predict=JM.Jx(Gains.flatten())

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
    

class ClassLM():
    def __init__(self,DATA,SM):
        JM=ClassJacobian(SM)
        JM.setDATA(DATA)
        JM.CalcKernelMatrix(DATA,iAnt)

        Jacob= JM.CalcJacobianAntenna(Gains,10)


        self.JHJ=np.dot(J.T.conj(),J)
        self.JHJinv=ModLinAlg.invSVD(self.JHJ)
