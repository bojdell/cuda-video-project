CC=gcc
CFLAGS=-c -Wall
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE         = ppm
OBJ         = ppm.o


default: $(EXE)

ppm.o: ppm.cu ppmKernel.cu
	$(NVCC) -c -o $@ ppm.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
