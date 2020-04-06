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
import getpass
import os
from killMS.Other import ModColor

import subprocess

from killMS import __version__


def report_version():
    # perhaps we are in a github with tags; in that case return describe
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        # work round possible unavailability of git -C
        result = subprocess.check_output('cd %s; git describe --tags' % path, shell=True, stderr=subprocess.STDOUT, universal_newlines=True).rstrip()
    except subprocess.CalledProcessError:
        result = None

    if result is not None and 'fatal' not in result:
        # will succeed if tags exist
        return result
    else:
        # perhaps we are in a github without tags? Cook something up if so
        try:
            result = subprocess.check_output('cd %s; git rev-parse --short HEAD' % path, shell=True, stderr=subprocess.STDOUT, universal_newlines=True).rstrip()
        except subprocess.CalledProcessError:
            result = None
        if result is not None and 'fatal' not in result:
            return __version__+'-'+result
        else:
            # we are probably in an installed version
            return __version__

def print_logo():

    #os.system('clear')
                                                       
    version=report_version()

    print("""       __        _   __   __   ____    ____   ______     """)
    print("""      [  |  _   (_) [  | [  | |_   \  /   _|.' ____ \    """)
    print("""       | | / ]  __   | |  | |   |   \/   |  | (___ \_|   """)
    print("""       | '' <  [  |  | |  | |   | |\  /| |   _.____`.    """)
    print("""       | |`\ \  | |  | |  | |  _| |_\/_| |_ | \____) |   """)
    print("""      [__|  \_][___][___][___]|_____||_____| \______.'   """)
    print("""             This is version : %s""" %ModColor.Str(version))
    print("""                                                        """)

