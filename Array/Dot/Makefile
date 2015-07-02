

all: dotSSE.so

# ---- Link --------------------------- 
dotSSE.so:  dotSSE.o 
	gcc -fopenmp -pthread -shared  dotSSE.o -o dotSSE.so

# ---- gcc C compile ------------------
dotSSE.o:  dotSSE.c dotSSE.h
	gcc -lgomp -fopenmp -pthread -Ofast --fast-math  -c dotSSE.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC
