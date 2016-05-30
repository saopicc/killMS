#!/usr/bin/env python
import numpy as np
import timeit

print
print np.__file__

ss="a=np.complex128(np.random.rand(52894,30)+1j*np.random.rand(52894, 30)); b=a+1"
print "Dot1: ",timeit.timeit("np.dot(a.T.conj(),b)",number=1,setup="import numpy as np; %s"%ss)

ss+="; a.fill(0)"
print "Dot2: ",timeit.timeit("np.dot(a.T.conj(),a)",number=1,setup="import numpy as np; %s"%ss)


