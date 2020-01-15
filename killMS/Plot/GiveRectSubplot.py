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
import pylab
import numpy as np

class giveRectSubPlot():
    def __init__(self,nx,ny,fig=None,pos=(0.1,0.1,0.9,0.9),xylabel=(False,False)):
        
        x0,y0,x1,y1=pos
        Lx=x1-x0
        Ly=y1-y0
        self.x0=x0
        self.y0=y0
        
        mfx=0.03 #margin factor
        mfy=0.03
        lx=(Lx/float(nx))*(1.+mfx*(nx+1)/nx)**(-1)
        ly=(Ly/float(ny))*(1.+mfy*(ny+1)/ny)**(-1)
        ly0=ly#/2
        mx=mfx*lx
        my=mfy*ly
        self.mx=mx
        self.my=my
        self.lx=lx
        self.ly=ly
        self.ly0=ly0
        self.nx=nx
        self.ny=ny
        self.fig=fig
        self.xylabel=xylabel

    def giveRect(self,ii):
        j=ii/self.nx
        i=ii-j*self.nx

        x0=self.mx+i*(self.lx+self.mx)
        y0=self.my+j*(self.ly+self.my)

        return self.x0+x0,self.y0+y0,self.lx,self.ly0
    
    def give_axis(self,i):
        R=self.giveRect(i)
        ax=self.fig.add_axes(R)
        if not(self.xylabel[0]):
            ax.set_xticklabels([])
            ax.get_xaxis().set_visible(False)
        if not(self.xylabel[1]):
            ax.set_yticklabels([])
            ax.get_yaxis().set_visible(False)
        ax.set_visible("off")
        return ax


    # def giveRect(self,ii,Mode="A"):
    #     j=ii/self.nx
    #     i=ii-j*self.nx
    #     print i,j
    #     if Mode=="A":
    #         x0=self.mx+i*(self.lx+self.mx)
    #         y0=self.my+j*(self.ly+self.my)
    #     else:
    #         x0=self.mx+i*(self.lx+self.mx)
    #         y0=self.my+j*(self.ly+self.my)+self.ly0

    #     return self.x0+x0,self.y0+y0,self.lx,self.ly0
    
    # def give_axis(self,i,Mode="A"):
    #     if type(i)==tuple:
    #         i,Mode=i

    #     R=self.giveRect(i,Mode)
    #     print i,Mode,R
    #     ax=self.fig.add_axes(R)
    #     return ax
