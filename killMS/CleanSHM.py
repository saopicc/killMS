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

import optparse
import sys

from DDFacet.Other import logger
from killMS.Array import NpShared
log=logger.getLogger("ClearSHM")

def read_options():
    desc="""CohJones Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* SHM")
    group.add_option('--ID',help='ID of ssared memory to be deleted, default is %default',default=None)
    opt.add_option_group(group)
    options, arguments = opt.parse_args()
    
    return options


if __name__=="__main__":
    options = read_options()
    print>>log, "Clear shared memory"
    if options.ID!=None:
        NpShared.DelAll(options.ID)
    else:
        NpShared.DelAll()

