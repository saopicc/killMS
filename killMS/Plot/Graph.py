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
from Plot import GiveRectSubplot
from . import GiveNXNYPanels

def test():

    G=ClassMplWidget(4)

    for i in range(4):
        G.subplot(i)
        G.imshow(interpolation="nearest",aspect="auto",origin='lower')#,extent=(-3,3,-3,3))
        #G.set_title("caca%i"%i)
        G.text(0,0,"caca%i"%i)
        G.draw()

    # G.plot(ls="",marker="+",color="red")
    # G.subplot(0)
    # G.plot()
    # G.subplot(0)
    # G.set_xylims((-3,3),(-3,3))
    # G.subplot(1)
    # G.set_xylims((-3,3),(-3,3))

    for i in range(100):
        for ii in range(4):
            G.subplot(ii)
            A=np.random.randn(100,100)
            G.imshow(A)
            G.text(0,0,"caca%i"%i)
        G.draw()

    # G2=ClassMplWidget(W2,1,2)
    





class ClassMplWidget():
    def __init__(self,Ntot,nx=None,ny=None,TBar=True,NPlots=None,SaveName="Fig",**kwargs):

        nx,ny=GiveNXNYPanels.GiveNXNYPanels(Ntot)
        self.CounterSave=0
        self.SaveName=SaveName

        #######
        #pylab.ion()
        #self.fig=pylab.figure(1,figsize=(15,7))
        self.fig=pylab.figure(figsize=(15,7))
        pylab.clf()
        self.mplwidget=self.fig.add_subplot(111)
        self.mplwidget.axes.set_visible(False)
        self.mplwidget.axes.set_axis_bgcolor("white")
        self.fig.patch.set_facecolor("white")
        #rect=0,0,1,1
        self.axisbg="white"
        self.mplwidget.axes.set_axis_bgcolor(self.axisbg)
        self.fig.patch.set_facecolor(self.axisbg)
        
        self.DicoAxis={}
        self.NPlots=NPlots
        if NPlots==None:
            self.NPlots=nx*ny
        self.nx=nx
        self.ny=ny
        self.CoordMachine=GiveRectSubplot.giveRectSubPlot(nx,ny,self.fig,pos=(0.05,0.05,0.95,0.95))
        self.ShowTicks=True


    def setSaveName(self,name):
        self.CounterSave=0
        self.SaveName=name

        


    def subplot(self,iAx,share=-1,draw=True,**kwargs):
        self.CurrentIAx=iAx
        if not(iAx in self.DicoAxis.keys()):
            self.DicoAxis[iAx]={}
            if share!=-1:
                ax=self.DicoAxis[share]["ax"]
                self.DicoAxis[iAx]["ax"]=self.mplwidget.figure.add_subplot(self.nx,self.ny,iAx+1,axisbg=self.axisbg,sharex=ax,sharey=ax)
            else:
                if self.CoordMachine==None:
                    self.DicoAxis[iAx]["ax"]=self.mplwidget.figure.add_subplot(self.nx,self.ny,iAx+1,axisbg=self.axisbg)
                else:
                    self.DicoAxis[iAx]["ax"]=self.CoordMachine.give_axis(iAx,**kwargs)

            if not(self.ShowTicks):
                ax=self.DicoAxis[iAx]["ax"]
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            self.DicoAxis[self.CurrentIAx]["CurrentIIm"]=-1
            self.DicoAxis[self.CurrentIAx]["CurrentIPlot"]=-1
            self.DicoAxis[self.CurrentIAx]["NLines"]=0

            if draw:
                #self.mplwidget.draw()
                pylab.draw()
        self.CurrentAx=self.DicoAxis[iAx]["ax"]
        self.CurrentIIm=self.DicoAxis[self.CurrentIAx]["CurrentIIm"]
        self.CurrentIPlot=self.DicoAxis[self.CurrentIAx]["CurrentIPlot"]

    def addToolBar(self):
        self.navi_toolbar = NavigationToolbar(self.mplwidget, self.parent)#self)
        self.gridLayout_2.addWidget(self.navi_toolbar)

    def cla(self):
        try:
            self.CurrentAx.draw_artist(self.CurrentAx.patch)
            self.fig.canvas.update()
            if self.CurrentIIm!=-1:
                self.CurrentIIm=0
            if self.CurrentIPlot!=-1:
                self.CurrentIPlot=0
        except:
            pass
        

    class ClassStrValPixel():
        def __init__(self,Dico):
            self.ax=Dico["ax"]
            self.data=Dico["data"][::-1].T
            self.ax.format_coord = lambda x,y: self.StrValPixel(x,y)
            self.im=Dico["Im"][0]
        def toPix(self,x0,x1,xi,N):

            D=x1-x0
            d=D/float(N)
            x=(xi-x0)/d#/D

            if x<0: x=0
            if x>=N: x=N-1
            #log.print( (x0,x1,xi,N,"->",x))
            return int(x)

        def StrValPixel(self,x,y):
            ax=self.ax
            data=self.data
            #x0,x1=ax.get_xlim()
            #y0,y1=ax.get_ylim()
            x0,x1,y0,y1=self.im.get_extent()
            xi=self.toPix(x0,x1,x,data.shape[0])
            yi=self.toPix(y0,y1,y,data.shape[1])
            val= data[xi,yi]
            #log.print( ax.get_xlim())
            #log.print( (x,y,xi,yi,val))
            return "(%i,%i): %5.4f" % (xi,yi,val)
    

    def imshow(self,*args,**kwargs):
        if len(args)==0:
            data=np.random.randn(200,200)
            args=(data,)
        else:
            data=args[0]

        if self.CurrentIIm==-1:#not("Plot" in self.DicoAxis[self.CurrentIAx].keys()):
            #log.print( ModColor.Str("   Axis [%i]: new im: %i"%(self.CurrentIAx,self.CurrentIIm),"blue"))
            if (self.CurrentIIm==-1): self.CurrentIIm=0
            if not("Im" in self.DicoAxis[self.CurrentIAx].keys()): self.DicoAxis[self.CurrentIAx]["Im"]=[]
            self.DicoAxis[self.CurrentIAx]["data"]=data

            if not("vmin" in kwargs.keys()):
                self.DicoAxis[self.CurrentIAx]["lims"]=(data.min(),data.max())
                kwargs["vmin"],kwargs["vmax"]=self.DicoAxis[self.CurrentIAx]["lims"]
            else:
                self.DicoAxis[self.CurrentIAx]["lims"]=kwargs["vmin"],kwargs["vmax"]
            self.DicoAxis[self.CurrentIAx]["Im"].append(self.CurrentAx.imshow(data,**kwargs))
            ax=self.DicoAxis[self.CurrentIAx]["ax"]
            if not("extent" in kwargs.keys()):
                self.DicoAxis[self.CurrentIAx]["xlims"]=ax.get_xlim()
                self.DicoAxis[self.CurrentIAx]["ylims"]=ax.get_ylim()
            #self.mplwidget.draw()
            pylab.draw()

        if data==None:
            data=np.random.randn(200,200)
        self.DicoAxis[self.CurrentIAx]["data"]=data
        self.DicoAxis[self.CurrentIAx]["getPix"]=self.ClassStrValPixel(self.DicoAxis[self.CurrentIAx])
 
        if self.CurrentIIm==len(self.DicoAxis[self.CurrentIAx]["Im"]):
            self.CurrentIIm=0
        ThisPlot=self.DicoAxis[self.CurrentIAx]["Im"][self.CurrentIIm]
        self.DicoAxis[self.CurrentIAx]["CurrentIIm"]=self.CurrentIIm
        self.CurrentIIm+=1

        #self.updateThread2(ThisPlot,data)
        ThisPlot.set_data(data)
        self.update(ThisPlot)

    def plot(self,*args,**kwargs):
        T=ClassTimeIt.ClassTimeIt("ColGraph.plot")
        #if self.CurrentIPlot==-1:#not("Plot" in self.DicoAxis[self.CurrentIAx].keys()):
        if len(args)==0:
            x,y=[0],[0]
            xmin,xmax=-3,3
            ymin,ymax=-3,3
            args=(x,y)
        else:
            x,y=args[0],args[1]
            xmin,xmax=x.min(),x.max()
            ymin,ymax=y.min(),y.max()
        xlim=xmin,xmax
        ylim=ymin,ymax
        #T.timeit("stuf0")
        if (self.DicoAxis[self.CurrentIAx]["NLines"]==0)|(self.CurrentIPlot>=self.DicoAxis[self.CurrentIAx]["NLines"]):
            #log.print( ModColor.Str("   new line: %i"%self.CurrentIPlot,"green"))
            #log.print( ModColor.Str("   Axis [%i]: new plot: %i"%(self.CurrentIAx,self.CurrentIPlot),"green"))
            self.DicoAxis[self.CurrentIAx]["NLines"]+=1
            if (self.CurrentIPlot==-1): self.CurrentIPlot=0
            if not("Plot" in self.DicoAxis[self.CurrentIAx].keys()): self.DicoAxis[self.CurrentIAx]["Plot"]=[]
            #T.timeit("stuf1.0")
             
            self.DicoAxis[self.CurrentIAx]["data"]=x,y
            if not("xlims" in self.DicoAxis[self.CurrentIAx].keys()):
                self.DicoAxis[self.CurrentIAx]["xlims"]=xlim
                self.DicoAxis[self.CurrentIAx]["ylims"]=ylim
            #T.timeit("stuf1.1")
            xlim=self.DicoAxis[self.CurrentIAx]["xlims"]
            ylim=self.DicoAxis[self.CurrentIAx]["ylims"]
            if not("Im" in self.DicoAxis[self.CurrentIAx].keys()): self.DicoAxis[self.CurrentIAx]["lims"]=None,None
            
            self.DicoAxis[self.CurrentIAx]["Plot"].append(self.CurrentAx.plot(*args,**kwargs)[0])
            #self.draw()
            #T.timeit("stuf1.2")
            #self.set_xylims(xlim,ylim)
            #T.timeit("stuf1.3")
 
            #self.DicoAxis[self.CurrentIAx]["ax"].set_xlim(xlim)
            #self.DicoAxis[self.CurrentIAx]["ax"].set_ylim(ylim)
#        if data==None:
#            data=np.random.randn(200,200)


        #if self.CurrentIPlot==len(self.DicoAxis[self.CurrentIAx]["Plot"]):
        #    self.CurrentIPlot=0
        ThisPlot=self.DicoAxis[self.CurrentIAx]["Plot"][self.CurrentIPlot]
        self.CurrentIPlot+=1
        #log.print( "   CurrentIPlot: %i"%self.CurrentIPlot)
        self.DicoAxis[self.CurrentIAx]["CurrentIPlot"]=self.CurrentIPlot
        #self.updateThread2(ThisPlot,data)

        #T.timeit("stuf2")
        ThisPlot.set_xdata(x)
        ThisPlot.set_ydata(y)
        #T.timeit("setdata2")
        self.update(ThisPlot)
        #T.timeit("update")


    def text(self,*args,**kwargs):

        if len(args)==0:
            x,y=0,0
            txt=""
        else:
            x,y=args[0],args[1]
            txt=args[2]

        if not("Text" in self.DicoAxis[self.CurrentIAx].keys()): 
            self.DicoAxis[self.CurrentIAx]["Text"]=self.CurrentAx.text(*args,**kwargs)

        ThisPlot=self.DicoAxis[self.CurrentIAx]["Text"]
        self.update(ThisPlot)



    def update(self,ThisPlot):
        self.CurrentAx.draw_artist(ThisPlot)

    def savefig(self):
        filename="%s_%4.4i.png"%(self.SaveName,self.CounterSave)
        #log.print( "   Saving figure in %s"%filename)
        self.fig.savefig(filename,dpi=100)
        #self.CounterSave+=1

    def draw(self):
        self.fig.canvas.update()
        self.fig.canvas.flush_events()

        

    def updateThread2(self,ThisPlot,data):
        self.threadPool = []
        self.threadPool.append( GenericThread(self.update,ThisPlot,data) )
        self.threadPool[len(self.threadPool)-1].start()

    def updateThread(self,ThisPlot,data):
        thread=AThread()
        thread.set(self,ThisPlot,data)
        thread.start()
        #thread.wait()
        #thread.terminate()
        #del(thread)

        
    def set_clim(self,axis,vmin,vmax):
        if not("Im" in self.DicoAxis[axis].keys()): return
        if self.CurrentIIm==len(self.DicoAxis[axis]["Im"]):
            self.CurrentIIm=0
        log.print( ("set_clim",vmin,vmax))
        ThisPlot=self.DicoAxis[axis]["Im"][self.CurrentIIm]
        ThisPlot.set_clim(vmin,vmax)
        self.DicoAxis[axis]["lims"]=vmin,vmax
        if vmin==None:
            log.print( vmin)
            return
        self.update(ThisPlot)

        if "Plot" in self.DicoAxis[axis]:
            for ThisPlot in self.DicoAxis[axis]["Plot"]:
                ThisPlot=self.DicoAxis[axis]["Plot"][self.CurrentIIm]
                self.update(ThisPlot)

        self.draw()
        #self.fig.canvas.flush_events()
        #self.mplwidget.draw()

    def set_extent(self,axis,extent):
        if not("Im" in self.DicoAxis[axis].keys()): return
        if self.CurrentIIm==len(self.DicoAxis[axis]["Im"]):
            self.CurrentIIm=0
        ThisPlot=self.DicoAxis[axis]["Im"][self.CurrentIIm]
        ThisPlot.set_extent(extent)

    def set_xylims(self,xlim,ylim):
        self.DicoAxis[self.CurrentIAx]["ax"].set_xlim(xlim)
        self.DicoAxis[self.CurrentIAx]["ax"].set_ylim(ylim)
        self.mplwidget.draw()

    def set_title(self,txt):
        ax=self.DicoAxis[self.CurrentIAx]["ax"]
        ax.set_title(txt)
        pylab.draw()

        

        


        
