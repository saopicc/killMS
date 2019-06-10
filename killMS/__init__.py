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

import pkg_resources
try:
    __version__ = pkg_resources.require("killMS")[0].version
except pkg_resources.DistributionNotFound:
    __version__ = "dev"

import os
# if not defined define.. this is a stopgap to support current workflows
# the better thing to do is not rely on environment variables and just use the
# path to the installed package
if "KILLMS_DIR" not in os.environ:
    os.environ["KILLMS_DIR"] = __path__[0]