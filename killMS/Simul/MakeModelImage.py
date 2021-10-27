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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pyrap.images import image
import numpy as np
from . import MakeClusterCat
import os

TemplateImage="TestDeconv.dirty.fits"#Clean.SS2.app.residual.fits"
#TemplateImage="Test_deconv.Corr.GA.dirty.fits"
OutFile="ModelImage"
Nc=0

class ClassMakeModelImage():
    def __init__(self):

        pass

    def main(self):
        self.MakeModelImage()
        self.WriteBBSCat()

    def WriteBBSCat(self):
        OutFileTXT=OutFile+".txt"
        MakeClusterCat.WriteBBSCat(OutFileTXT,self.Cat)
        sExec="MakeModel.py --SkyModel=%s --NCluster=%i --CMethod=4 --DoPlot=0"%(OutFileTXT,Nc)
        os.system(sExec)


    def MakeModelImage(self):
        Ns=5
        im=image(TemplateImage)
        im.saveas("MODEL")
        im=image("MODEL")
        IM=im.getdata()
        IM.fill(0)

        nch,npol,nx,_=IM.shape
        
        dx=int(nx/10.)
        indx,indy=np.mgrid[dx:nx-dx:Ns*1j,dx:nx-dx:Ns*1j]

        #indx,indy=np.mgrid[0:nx:Ns*1j,0:nx:Ns*1j]

        indx=np.int64(indx.flatten())
        indy=np.int64(indy.flatten())

        # indx=indx[0:1]
        # indy=indy[0:1]

        indx+=np.int64(np.random.randn(indx.size)*50)
        indy+=np.int64(np.random.randn(indx.size)*50)
        # #stop
        


        # indx,indy=np.int64(np.random.rand(Ns)*nx),np.int64(np.random.rand(Ns)*nx)

        # indx,indy=nx/4+np.int64(np.random.rand(Ns)*nx/2),nx/4+np.int64(np.random.rand(Ns)*nx/2)
        # S=np.random.rand(indx.size)*100
        # IM[0,0,indy,indx]=IM[0,0,indy,indx]+S**1.5

        # R=float(nx)/2.
        # indx,indy=nx/2+np.int64(np.random.randn(Ns)*R),nx/2+np.int64(np.random.randn(Ns)*R)

        Cx=((indx>=0)&(indx<nx))
        Cy=((indy>=0)&(indy<nx))
        ind=np.where(Cx&Cy)[0]
        indx=indx[ind]
        indy=indy[ind]
        
        S=np.random.rand(indx.size)*100
        #S.fill(100)

        # indx[0]=6000
        # indy[0]=1000
        # Smax=np.max(S)
        # S[0]=5*Smax
        # indx[1]=3000
        # indy[1]=4000
        # S[1]=50
        # indx[2]=3000
        # indy[2]=4020
        # S[2]=30


        # #S[0]=10000
        # S=S[0:1]
        # S.fill(1)
        # indx=indx[0:1]
        # indy=indy[0:1]
        # indx[0]=200
        # indy[0]=100

        #IM[0,0,indy,indx]=S**1.5
        IM[0,0,indy,indx]=S


        

        # indy,indx=np.where(IM[0,0]!=0)
        Ns=np.count_nonzero(IM)
        print("Number of sources",indx.size)
        Alpha=np.random.rand(Ns)*2-1.
        #Alpha[0]=0
        Alpha.fill(0)
        im.putdata(IM)
        im.tofits(OutFile+".fits")

        
        Cat=np.zeros((Ns,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                  ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                  ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                  ('Gmaj',np.float),('Gangle',np.float),("Select",np.int)])
        Cat=Cat.view(np.recarray)
        


        pol,freq,decc,rac=im.toworld((0,0,0,0))
        
       
        for iSource in range(Ns):
            x_iSource,y_iSource=indx[iSource],indy[iSource]

            _,_,dec_iSource,ra_iSource=im.toworld((0,0,y_iSource,x_iSource))
            Cat.ra[iSource]=ra_iSource
            Cat.dec[iSource]=dec_iSource
            Flux=IM[0,0,y_iSource,x_iSource]
            Cat.I[iSource]=Flux
            Cat.alpha[iSource]=Alpha[iSource]

        # import pylab
        # pylab.clf()
        # pylab.scatter(Cat.ra,Cat.dec)
        # pylab.draw()
        # pylab.show()

        self.Cat=Cat


def main():
    CMI=ClassMakeModelImage()
    CMI.main()
    


if __name__=="__main__":
    main()
