#!/usr/bin/env python
"""
killMS, a package for calibration in radio interferometry.
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import sys
from killMS.Other import MyPickle
from killMS.Other import logo
from killMS.Other import ModColor
from DDFacet.Other import logger
import os
log=logger.getLogger("killMS")
log.logger.setLevel(logger.logging.CRITICAL)

sys.path=[name for name in sys.path if not(("pyrap" in name)&("/usr/local/lib/" in name))]

# test

#import numpy
#print numpy.__file__
#import pyrap
#print pyrap.__file__
#stop


if "nocol" in sys.argv:
    print("nocol")
    ModColor.silent=1
if "nox" in sys.argv:
    import matplotlib
    matplotlib.use('agg')
    print(ModColor.Str(" == !NOX! =="))

import time
import os
import numpy as np
import pickle
from itertools import product as ItP

NameSave="last_plotSols.obj"
def read_options():

    desc="""killMS Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SolsFile',help='Input Solutions list [no default]',default='')
    group.add_option('--DoResid',type="int",help='No [no default]',default=-1)
    group.add_option('--PlotMode',type='str',help=' [no default]',default="AP")
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    f = open(NameSave,"wb")
    pickle.dump(options,f)
    
import pylab

import numpy as np

def GiveNXNYPanels(Ns,ratio=800/500):
    nx=int(round(np.sqrt(Ns/ratio)))
    ny=int(nx*ratio)
    if nx*ny<Ns: ny+=1
    return nx,ny

from killMS.Array import ModLinAlg

def NormMatrices(G):
    print("no norm")
    return G
    nt,nch,na,_,_=G.shape

    for iChan,it in ItP(range(nch),range(nt)):
        Gt=G[it,iChan,:,:]
        u,s,v=np.linalg.svd(Gt[0])
        # #J0/=np.linalg.det(J0)
        # J0=Gt[0]
        # JJ=np.dot(J0.T.conj(),J0)
        # sqJJ=ModLinAlg.sqrtSVD(JJ)
        # sqJJinv=ModLinAlg.invSVD(JJ)
        # U=np.dot(J0,sqJJinv)
        U=np.dot(u,v)
        for iAnt in range(0,na):
            Gt[iAnt,:,:]=np.dot(U.T.conj(),Gt[iAnt,:,:])
            #Gt[iAnt,:,:]=np.dot(np.dot(u,Gt[iAnt,:,:]),v.T.conj())
            #Gt[iAnt,:,:]=np.dot(Gt[iAnt,:,:],J0)
    return G


def main(options=None):
    

    if options==None:
        f = open(NameSave,'rb')
        options = pickle.load(f)



    
    FilesList=options.SolsFile.split(",")


    
    if not os.path.isdir("png"):
        os.system("mkdir -p png")

    LSols=[]
    nSol=len(FilesList)
    t0=None
    for FileName in FilesList:
        
        if "npz" in FileName:
            SolsDico=np.load(FileName)
            Sols=SolsDico["Sols"]
            StationNames=SolsDico["StationNames"]
            ClusterCat=SolsDico["ClusterCat"]
            Sols=Sols.view(np.recarray)
            nt,nch,na,nd,_,_=Sols.G.shape
        elif "h5" in FileName:
            import tables
            H5=tables.openFile(FileName)
            npol, nch, nd, na, nchan, nt=H5.root.sol000.amplitude000.val.shape
            GH5=H5.root.sol000.amplitude000.val[:]*np.exp(1j*H5.root.sol000.phase000.val[:])
            Times=H5.root.sol000.amplitude000.time[:]
            StationNames=H5.root.sol000.antenna[:]["name"]
            H5.close()
            

            Sols=np.zeros((nt,),dtype=[("t0",np.float64),
                                       ("t1",np.float64),
                                       ("G",np.complex64,(nch,na,nd,2,2))])
            Sols=Sols.view(np.recarray)
            dt=np.median(Times[1::]-Times[0:-1])
            Sols.t0=Times-dt/2.
            Sols.t1=Times+dt/2.
            for iTime in range(nt):
                for iDir0,iDir1 in zip(range(3),range(3)):#[0,2,1]):#range(nd):
                    for iAnt in range(na):
                        for ipol in range(4):
                            Sols.G[iTime,iAnt,iDir0].flat[ipol]=GH5[ipol,iDir1,iAnt,0,iTime]

            
        ind=np.where(Sols.t1!=0)[0]
        Sols=Sols[ind]
        tm=(Sols.t1+Sols.t0)/2.
        if t0==None:
            t0=tm[0]
        tm-=t0
        Sols.t0=tm
        nx,ny=GiveNXNYPanels(na)
        LSols.append(Sols)

        # LSols=[LSols[0]]
        # nSol=1

    # diag terms
    Lls=["-",":",":"]
    Lcol0=["black","black","blue"]
    Lcol1=["gray","gray","red"]
    Lalpha0=[1,1,1]
    Lalpha1=[0.5,0.5,0.5]

    # Lls=["-","-",":"]
    # Lcol0=["black","blue","blue"]
    # Lcol1=["gray","red","red"]
    # Lalpha0=[1,0.5,1]
    # Lalpha1=[0.5,0.5,0.5]


    # off-diag terms
    Lls_off=Lls#["-","--",":"]
    Lcol0_off=Lcol0#["black","black","blue"]
    Lcol1_off=Lcol1#["gray","gray","red"]
    
    if options.DoResid!=-1:
        Sresid=LSols[1].copy()
        LSols.append(Sresid)

    DirList=range(nd)

    #DirList=[np.where(ClusterCat["SumI"]==np.max(ClusterCat["SumI"]))[0][0]]
    #print DirList
    nt,nch,na,nd,_,_=Sols.G.shape

    for iDir in DirList:
        iAnt=0
        for iSol in range(nSol):
            Sols=LSols[iSol]
            G=Sols.G[:,:,:,iDir,:,:]
            Sols.G[:,:,:,iDir,:,:]=NormMatrices(G)
            

            
    tm=(LSols[0].t0+LSols[0].t1)/2.
    tm-=tm[0]
    tm/=60.

    fig=pylab.figure(0,figsize=(13,8))
    for iTime in range(nt):
        print("%i/%i"%(iTime,nt))
        ampMax=1.5*np.max(np.median(np.abs(LSols[0].G),axis=0))
        if options.PlotMode=="AP":
            op0=np.abs
            op1=np.angle
            ylim0=0,ampMax
            ylim1=-np.pi,np.pi
            PlotDiag=[True,False]
        elif options.PlotMode=="ReIm":
            op0=np.real
            op1=np.imag
            ylim0=-ampMax,ampMax
            ylim1=-ampMax,ampMax
            PlotDiag=[True,True]
        elif options.PlotMode=="A":
            op0=np.abs
            op1=None
            ylim0=0,ampMax
            PlotDiag=[True]
        elif options.PlotMode=="P":
            op0=np.angle
            op1=None
            ylim0=-np.pi,np.pi
            PlotDiag=[False]

        # L_ylim0=(0,1.5*np.max(np.median(np.abs(LSols[0].G[:,:,:,iDir,:,:]),axis=0)))

        # if options.DoResid!=-1:
        #     LSols[-1].G[:,:,iDir,:,:]=LSols[1].G[:,:,iDir,:,:]-LSols[0].G[:,:,iDir,:,:]
        #     nSol+=1
        # marker="."

        lmax=1.5*np.max([np.max(np.abs(ClusterCat["l"])),np.max(np.abs(ClusterCat["m"]))])
        Npix=101
        lgrid,mgrid=np.mgrid[-lmax:lmax:1j*Npix,-lmax:lmax:1j*Npix]
        lgrid,mgrid=lgrid.reshape((-1,1)),mgrid.reshape((-1,1))
        d=np.sqrt((lgrid-ClusterCat["l"].reshape((1,-1)))**2+(mgrid-ClusterCat["m"].reshape((1,-1)))**2)
        inode=np.argmin(d,axis=1)
        

        A=np.zeros((Npix,Npix),np.complex64)
        pylab.clf()

        for i in range(nx):
            for j in range(ny):
                if iAnt>=na:continue
                if iAnt>=1:
                    ax=pylab.subplot(nx,ny,iAnt+1,sharex=axRef,sharey=axRef)
                else:
                    axRef=pylab.subplot(nx,ny,iAnt+1)
                    ax=axRef

                if op1!=None: ax2 = ax.twinx()

                pylab.title(StationNames[iAnt], fontsize=9)
                for iChan in range(nch):
                    Sols=LSols[iSol]
                    #G=Sols.G[:,iChan,:,iDir,:,:]
                    G=Sols.G[iTime,iChan,:,:,:,:]
                    J=G[iAnt,:,:,:]
                    A.flat[:]=J[:,0,0][inode]
                        
                    #ax.scatter(ClusterCat["l"],ClusterCat["m"],c=op0(J[:,0,0]),vmin=0,vmax=ampMax)
                    ax.imshow(op0(A),vmin=0,vmax=ampMax)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # if op1!=None:
                    #     ax2.plot(Sols.t0,op1(J[:,1,1]),color=Lcol1[iSol],alpha=Lalpha1[iSol],ls=Lls[iSol],marker=marker)
                    #     ax2.plot(Sols.t0,op1(J[:,0,0]),color=Lcol1[iSol],alpha=Lalpha1[iSol],ls=Lls[iSol],marker=marker)
                    #     if PlotDiag[1]:
                    #         ax2.plot(Sols.t0,op1(J[:,0,1]),color=Lcol1_off[iSol],alpha=Lalpha1[iSol],ls=Lls_off[iSol],marker=marker)
                    #         ax2.plot(Sols.t0,op1(J[:,1,0]),color=Lcol1_off[iSol],alpha=Lalpha1[iSol],ls=Lls_off[iSol],marker=marker)
                    #     ax2.set_ylim(ylim1)
                    #     ax2.set_xticks([])
                    #     ax2.set_yticks([])
                    #     #print StationNames[iAnt]

                iAnt+=1
        pylab.suptitle('Time since start %6.2f minutes'%(tm[iTime]))#L_ylim0)))
        #pylab.tight_layout(pad=3., w_pad=0.5, h_pad=2.0)
        pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        fig.savefig("png/%5.5i.png"%iTime)
        #time.sleep(1)
        iAnt=0

    OutFile="animation.gif"
    log.print("Creating %s"%OutFile)
    os.system("convert -delay 10 -loop 0 png/*.png %s"%OutFile)


def driver():
    read_options()
    f = open(NameSave,'rb')
    options = pickle.load(f)

    main(options=options)

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()