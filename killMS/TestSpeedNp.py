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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import timeit

print()
print(np.__file__)

ss="a=np.complex128(np.random.rand(52894,30)+1j*np.random.rand(52894, 30)); b=a+1"
print("Dot1: ",timeit.timeit("np.dot(a.T.conj(),b)",number=1,setup="import numpy as np; %s"%ss))

ss+="; a.fill(0)"
print("Dot2: ",timeit.timeit("np.dot(a.T.conj(),a)",number=1,setup="import numpy as np; %s"%ss))


