#!/usr/bin/env python

import optparse
import sys
from Other import MyPickle
from Other import ModColor

sys.path=[name for name in sys.path if not(("pyrap" in name)&("/usr/local/lib/" in name))]


import numpy as np
from Data import ClassMS
from Other import MyLogger
log=MyLogger.getLogger("MSTools")

def read_options():
    desc="""CohJones Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--ms',help='Input MS to draw [no default]',default='')
    group.add_option('--Operation',help='BACKUP: create backup | COPY: copy one column to another Default is %default',default='BACKUP')
    group.add_option('--TChunk',help='Time chunk in hours, default is %default',default=15)
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* BACKUP options")
    group.add_option('--Col',help='Column to backup, default is %default',default='CORRECTED_DATA')
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* COPY options")
    group.add_option('--InOutCol',help='Column to copy, default is %default',default='CORRECTED_DATA_BACKUP,CORRECTED_DATA')
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    options.TChunk=float(options.TChunk)
    
    return options

def PutBackupCols(options):
    MS=ClassMS.ClassMS(options.ms,DoReadData=False,TimeChunkSize=options.TChunk)
    MS.PutBackupCol(incol=options.Col)
    

def Copy(options=None):
    MS=ClassMS.ClassMS(options.ms,DoReadData=False,TimeChunkSize=options.TChunk)
    In,Out=options.InOutCol.split(",")
    MS.CopyCol(In,Out)



if __name__=="__main__":
    options = read_options()

    if options.Operation=="BACKUP":
        PutBackupCols(options)
    elif options.Operation=="COPY":
        Copy(options)

