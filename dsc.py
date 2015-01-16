#!/usr/bin/python

import os
import sys
from reformat import *

if __name__=="__main__":


    imin=sys.argv[1]
    imin=os.path.realpath(sys.argv[1])

    if len(sys.argv)==2:
        imout=reformat(imin,LastSlash=False)+".fits"
        


    os.system("rm -f %s"%imout)
    strexec="image2fits in=%s out=%s; ds9 %s"%(imin,imout,imout)
    os.system(strexec)




