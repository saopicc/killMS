#!/usr/bin/env python
import numpy as np
from pyrap.tables import table
#import pylab
import optparse
import time
#import managecolumns
from killMS.Other import MyLogger
log=MyLogger.getLogger("ClipCal")
MyLogger.itsLog.logger.setLevel(MyLogger.logging.CRITICAL)
SaveFile="ClipCal.last"
import pickle

class ClassClipMachine():
    def __init__(self,MSName,Th=20.,ColName="CORRECTED_DATA",SubCol=None,WeightCol="IMAGING_WEIGHT"):
        self.MSName=MSName
        t=self.t=table(MSName,ack=False)
        print>>log,"Reading visibility column %s from %s"%(ColName,MSName)
        vis=t.getcol(ColName)
        if SubCol is not None:
            print>>log,"  Subtracting column %s"%SubCol
            vis1=t.getcol(SubCol)
            vis-=vis1

        print>>log,"  Reading flags"
        flag=t.getcol("FLAG")
        print>>log,"  Zeroing flagged data"
        vis[flag==1]=0
        self.WeightCol=WeightCol

        print>>log,"  Reading weights column %s"%WeightCol
        W=t.getcol(WeightCol)
        t.close()

        self.vis=vis
        self.flag=flag
        self.W=W
        self.Th=Th

    def ClipWeights(self):
        vis=self.vis
        flag=self.flag
        W=self.W

        nrow,nch,npol=vis.shape

        print>>log,"Fraction of zero-weighted visibilities previously non-flagged"
        for ch in range(nch):
            print>>log,"Channel = %i"%ch
            for pol in range(npol):
                f=(flag[:,ch,pol]==0)

                AbsVis=np.abs(vis[:,ch,pol])
                AbsVis_s=AbsVis[f]
                if AbsVis_s.size==0:
                    print>>log,"  All data is flagged - skipping..."
                    continue
                
                MAD=np.median(AbsVis_s)
                std=1.48*MAD
                M=(AbsVis>self.Th*std)
                ind=np.where(M)[0]
                W[ind]=0

                Ms=(AbsVis[f]>self.Th*std)
                nfg=np.count_nonzero(Ms)
                frac=nfg/float(Ms.size)

                print>>log,"  pol#%i %.7f%% [n=%i, <rms> = %f]"%(pol,frac*100.,nfg,std)

        print>>log,"Writting %s in %s"%(self.WeightCol,self.MSName)
        t=table(self.MSName,readonly=False,ack=False)
        t.putcol(self.WeightCol,W)
        t.close()


def read_options():
    desc=""" Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--MSName',help='Input MS to draw [no default]',default='')
    group.add_option('--Th',help='Level above which clip (in sigma. Default is <MSName>.%default.',type=float,default=20.)
    group.add_option('--ColName',help='Input column. Default is %default',default='CORRECTED_DATA')
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
        CM=ClassClipMachine(MSName,options.Th,ColName,SubCol=SubCol,WeightCol="IMAGING_WEIGHT")
        CM.ClipWeights()
        if len(LMSName)>1:
            print>>log,"=========================================="


if __name__=="__main__":
    options=read_options()
    f = open(SaveFile,'rb')
    options = pickle.load(f)
    main(options=options)
