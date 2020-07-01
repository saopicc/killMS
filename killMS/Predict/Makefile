
all: _predict3.so

# ---- Link --------------------------- 
_predict3.so:  predict.o 
	gcc -fopenmp -pthread -shared  predict.o -o _predict3.so

# ---- gcc C compile ------------------
predict.o:  predict.c predict.h
	gcc -lgomp -fopenmp -pthread -Ofast --fast-math  -c predict.c -I/usr/include/python3.6 -I/usr/lib/python3/dist-packages/numpy/core/include/numpy -fPIC
