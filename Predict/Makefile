

all: predict.so predict_np19.so


# ---- Link --------------------------- 
predict.so:  predict.o 
	gcc -fopenmp -pthread -shared  predict.o -o predict.so

# ---- gcc C compile ------------------
predict.o:  predict.c predict.h
#	gcc -lgomp -fopenmp -pthread -Ofast --fast-math  -c predict.c -I/usr/include/python2.7 -fPIC
	gcc -lgomp -fopenmp -pthread -Ofast --fast-math  -c predict.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC
#	gcc -lgomp -fopenmp -pthread -Ofast --fast-math  -c predict.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC

#	gcc -lgomp -fopenmp -pthread -Ofast -msse -msse2 -msse3 -msse4.2 -mssse3 -march=corei7 -mtune=corei7 --fast-math  -c predict.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -fPIC


# ---- Link --------------------------- 
predict_np19.so:  predict_np19.o 
	gcc -fopenmp -pthread -shared  predict_np19.o -o predict_np19.so

predict_np19.o:  predict_np19.c predict_np19.h
	gcc -lgomp -fopenmp -pthread -Ofast --fast-math  -c predict_np19.c -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy -I/home/cyril.tasse/bin/lib/python2.7/site-packages/numpy/core/include/numpy -fPIC
