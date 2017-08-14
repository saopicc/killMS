INC=$(shell python-config --includes) $(shell python -c "import numpy; print '-I' + numpy.get_include() + '/numpy'")

all: dotSSE.so

# remove compilation products
clean:
	rm -f *.o *.so


# ---- Link --------------------------- 
dotSSE.so:  dotSSE.o 
	gcc -shared  dotSSE.o -o dotSSE.so

# ---- gcc C compile ------------------
dotSSE.o:  dotSSE.c dotSSE.h
	gcc -O3 --fast-math  -c dotSSE.c $(INC) -fPIC
