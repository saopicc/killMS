#!/usr/bin/env python

import optparse
import sys
from killMS2.Other import MyPickle
from killMS2.Other import ModColor
from pyrap.tables import table
import glob

sys.path=[name for name in sys.path if not(("pyrap" in name)&("/usr/local/lib/" in name))]


import numpy as np
from killMS2.Data import ClassMS
from killMS2.Other import MyLogger
log=MyLogger.getLogger("MSTools")
import os

def read_options():
    desc="""CohJones Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--ms',help='Input MS to draw [no default]',default='')
    group.add_option('--Operation',help='BACKUP: create backup | COPY: copy one column to another | CasaCols: Put casa columns in MS. Default is %default',default='CasaCols')
    group.add_option('--TChunk',help='Time chunk in hours, default is %default',default=15)
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* BACKUP options")
    group.add_option('--Col',help='Column to backup, default is %default',default='CORRECTED_DATA')
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* COPY options")
    group.add_option('--InOutCol',help='Column to copy, default is %default',default='CORRECTED_DATA_BACKUP,CORRECTED_DATA')
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* SPLIT_SCAN options")
    group.add_option('--OutputMSList',type="str",help='OutputMSList, default is %default',default='MSList.txt')
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    options.TChunk=float(options.TChunk)
    
    return options

class MSTools():
    def __init__(self,options):
        self.options=options

    def PutBackupCols(self):
        options=self.options
        MS=ClassMS.ClassMS(options.ms,DoReadData=False,TimeChunkSize=options.TChunk)
        MS.PutBackupCol(incol=options.Col)
    
    def PutCasaCols(self):
        options=self.options
        MS=ClassMS.ClassMS(options.ms,DoReadData=False,TimeChunkSize=options.TChunk)
        MS.PutCasaCols()
        
    
    def Copy(self):
        options=self.options
        MS=ClassMS.ClassMS(options.ms,DoReadData=False,TimeChunkSize=options.TChunk)
        In,Out=options.InOutCol.split(",")
        MS.CopyCol(In,Out)
    
    
    def SplitSCAN_MS_List(self):
        options=self.options
        MSList=options.ms
        self.ListFiles=[]
        #self.ScansDir="SCANS"
        #os.system("mkdir -p %s"%self.ScansDir)

        ll=glob.glob(MSList)
        
        for MSn in ll:
            self.SplitSCAN_MS(MSn)

        OutputMSList=options.OutputMSList
        print "Writing list of MS in %s"%OutputMSList
        f = open(OutputMSList, 'w')
        for MSn in self.ListFiles:
            f.write('%s\n'%MSn)
        f.close()
        
        
    def SplitSCAN_MS(self,MSName):
        MSName=options.ms
        t=table(MSName,ack=False)
        ListScanID=sorted(list(set(list(t.getcol("SCAN_NUMBER")))))
        t.close()
        
        ID=0
        for ScanID in ListScanID:
            #MSOut="%s.SCAN_%4.4i.MS"%(self.ScansDir,MSName,ID)
            MSOut="%s.SCAN_%4.4i.MS"%(MSName,ID)
            ID+=1
            ss="taql 'SELECT FROM %s WHERE SCAN_NUMBER==%i GIVING %s'"%(MSName,ScanID,MSOut)
            print ss
            os.system(ss)
            self.ListFiles.append(MSOut)

        

if __name__=="__main__":
    options = read_options()
    MST=MSTools(options)

    if options.Operation=="BACKUP":
        MST.PutBackupCols()
    elif options.Operation=="CasaCols":
        MST.PutCasaCols()
    elif options.Operation=="COPY":
        MST.Copy()
    elif options.Operation=="SPLIT_SCAN":
        MST.SplitSCAN_MS_List()

