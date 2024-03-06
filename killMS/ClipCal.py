#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from pyrap.tables import table
#import pylab
import optparse
import time
#import managecolumns
from DDFacet.Other import logger
log=logger.getLogger("ClipCal")
# next line adapted for newer DDFacet versions
log.log_verbosity(logger.logging.CRITICAL)
SaveFile="ClipCal.last"
import pickle
from killMS.Data import ClassMS

class ClassClipMachine():
    def __init__(self,MSName,Th=20.,
                 ColName="CORRECTED_DATA",
                 SubCol=None,
                 WeightCol="IMAGING_WEIGHT",
                 ReinitWeights=False,
                 InFlagCol="FLAG",
                 #OutType="Weight"
    ):
        self.MSName=MSName




        t=self.t=table(MSName,ack=False)



        
        if WeightCol=="FLAG" and "FLAG_BACKUP_CLIPCAL" not in t.colnames():
            self.MS=ClassMS.ClassMS(self.MSName,
                                    Col=ColName,
                                    DoReadData=False)
            self.MS.AddCol("FLAG_BACKUP_CLIPCAL",LikeCol="FLAG")
            t.putcol("FLAG_BACKUP_CLIPCAL",t.getcol("FLAG"))

        log.print("Reading visibility column %s from %s"%(ColName,MSName))
        vis=t.getcol(ColName)
        if SubCol is not None:
            log.print("  Subtracting column %s"%SubCol)
            vis1=t.getcol(SubCol)
            vis-=vis1

        log.print("  Reading flags column: %s"%InFlagCol)
        flag=t.getcol(InFlagCol)
        _,nch,npol=flag.shape
        Lch=self.Lch=slice(None) # slice(726,727)
        
        for ch in range(nch)[Lch]:
            for ipol in range(npol):
                fs=flag[:,ch,ipol]
                ff=np.count_nonzero(fs)/fs.size
                log.print("[ch,pol, fracFlagged] = %i, %i, %f"%(ch,ipol,ff))
                
        log.print("  Zeroing flagged data")
        vis[flag==1]=0
        self.WeightCol=WeightCol

        nrow,nch,_=vis.shape
        if WeightCol in t.colnames():
            log.print("  Reading weights column %s"%WeightCol)
            print("!!!!!!!!!!!!!!!")
            W=t.getcol(WeightCol)
            if WeightCol=="FLAG":
                log.print("  ... these are flags - taking ones complement to get weights")
                W=1-W
                
            if ReinitWeights:
                log.print("  Initialise weights to one")
                W.fill(1)
                
        else:
            log.print("  Column %s not present, creating ones-like weight array"%WeightCol)
            W=np.ones((nrow,nch),np.float32)
            
        t.close()

        self.vis=vis
        self.flag=flag
        self.W=W
        self.Th=Th
        #self.OutType=OutType
        
    def ClipWeights(self):
        vis=self.vis
        flag=self.flag
        W=self.W

        nrow,nch,npol=vis.shape

        Lch=self.Lch
        
        log.print("Fraction of zero-weighted visibilities previously non-flagged")
        for ch in range(nch)[Lch]:
            log.print("Channel = %i"%ch)
            for pol in range(npol):
                f=(flag[:,ch,pol]==0)

                
                f1=(flag[:,ch,pol]==1)
                W[f1,ch]=0
                
                ws=W[:,ch]
                # frac0=np.count_nonzero(ws==0)/ws.size
                
                AbsVis=np.abs(vis[:,ch,pol])
                AbsVis_s=AbsVis[f]
                
                if AbsVis_s.size==0:
                    log.print("  All data is flagged - skipping...")
                    continue
                
                MAD=np.median(AbsVis_s)
                std=1.48*MAD
                M=(AbsVis>self.Th*std)
                ind=np.where(M)[0]
                W[ind,ch]=0

                Ms=(AbsVis[f]>self.Th*std)
                nfg=np.count_nonzero(Ms)
                frac=nfg/float(Ms.size)
                ntot=Ms.size

                ws=W[:,ch]
                # frac1=np.count_nonzero(ws==0)/ws.size
                # log.print("  pol#%i %.7f -> %.7f %% [n=%i/%i, <rms> = %f]"%(pol,frac0*100.,frac1*100.,nfg,ntot,std))


        t=table(self.MSName,readonly=False,ack=False)
        
        log.print("Writing %s in %s"%(self.WeightCol,self.MSName))

        if self.WeightCol not in t.colnames():
            log.print("Putting column %s in %s"%(self.WeightCol,self.MSName))
            self.MS=ClassMS.ClassMS(self.MSName,
                                    Col="DATA",
                                    DoReadData=False)
            self.MS.AddCol(self.WeightCol,ColDesc="IMAGING_WEIGHT")

        if self.WeightCol=="FLAG":
            #Wf=W.reshape((nrow,nch))*np.ones((1,1,npol))
            #W=Wf.astype(self.flag.dtype)
            W=(W==0)
            for ch in range(nch)[Lch]:
                for pol in range(npol):
                    w=W[:,ch,pol]
                    ff=np.count_nonzero(w)/w.size
                    log.print("[ch, pol, fracFlagged] = %i, %i, %f"%(ch,pol,ff))
                
            ff=np.count_nonzero(W)/W.size
            log.print("[fracFlaggedTotal] = %f"%(ff))
            
            
        t.putcol(self.WeightCol,W)
        
        # if self.OutType=="Weight":
        #     log.print("Writing %s in %s"%(self.WeightCol,self.MSName))
        #     self.MS.AddCol(self.WeightCol,ColDesc="IMAGING_WEIGHT")
        #     if self.WeightCol=="FLAG":
        #         Wf=W*np.ones((1,1,4))
        #         W=Wf.astype(self.flag.dtype)
        #     t.putcol(self.WeightCol,W)
        # elif self.OutType=="Flag":
        #     log.print("Writing %s in %s"%(self.WeightCol,self.MSName))
        #     if "FLAG_BACKUP_CLIPCAL" not in t.colnames():
        #         self.MS.AddCol("FLAG_BACKUP_CLIPCAL",LikeCol="FLAG")
        #         t.putcol("FLAG_BACKUP_CLIPCAL",t.getcol("FLAG"))
        #     if self.WeightCol=="FLAG":
        #         Wf=W*np.ones((1,1,4))
        #         W=Wf.astype(self.flag.dtype)
        #     t.putcol(self.WeightCol,W)
            
        t.close()


def read_options():
    desc=""" Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--MSName',help='Input MS to draw [no default]',default='')
    group.add_option('--Th',help='Level above which clip (in sigma. Default is <MSName>.%default.',type=float,default=20.)
    group.add_option('--ColName',help='Input column. Default is %default',default='CORRECTED_DATA')
    group.add_option('--WeightCol',help='Input column IMAGING_WEIGHT/FLAG. Default is %default',default='IMAGING_WEIGHT')
    group.add_option('--ReinitWeights',help='Input column. Default is %default',type=int,default=0)
    group.add_option('--InFlagCol',help='Input column (FLAG/FLAG_BACKUP_CLIPCAL). Default is %default',default="FLAG")
    opt.add_option_group(group)
    options, arguments = opt.parse_args()
    f = open(SaveFile,"wb")
    pickle.dump(options,f)
    return options

def main(options=None):
    
    if options==None:
        f = open(SaveFile,'rb')
        options = pickle.load(f)

    if ".txt" in options.MSName:
        LMSName = [ l.strip() for l in open(options.MSName).readlines() ]
    else:
        LMSName = [ options.MSName ]

    SubCol=None
    if "," in options.ColName:
        ColName,SubCol=options.ColName.split(",")
    else:
        ColName=options.ColName
        
    for MSName in LMSName:
        CM=ClassClipMachine(MSName,options.Th,ColName,SubCol=SubCol,WeightCol=options.WeightCol,
                            ReinitWeights=options.ReinitWeights,InFlagCol=options.InFlagCol)
        CM.ClipWeights()
        if len(LMSName)>1:
            log.print("==========================================")


def driver():
    options=read_options()
    f = open(SaveFile,'rb')
    options = pickle.load(f)
    main(options=options)

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()