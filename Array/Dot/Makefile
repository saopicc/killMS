all: dotSSE.so

# remove compilation products
clean:
	rm -f *.o *.so


# ---- Link --------------------------- 
dotSSE.so:  dotSSE.o 
	gcc -shared  dotSSE.o -o dotSSE.so

# ---- gcc C compile ------------------
dotSSE.o:  dotSSE.c dotSSE.h
	gcc -O3 --fast-math  -c dotSSE.c -I/usr/include/python2.7 -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy  -I/home/tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -I/home/tasse/bin/lib64/python2.7/site-packages/numpy-1.8.0-py2.7-linux-x86_64.egg/numpy/core/include/numpy -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -I/home/tasse/bin/lib64/python2.7/site-packages/numpy-1.8.0-py2.7-linux-x86_64.egg/numpy/core/include/numpy -fPIC
