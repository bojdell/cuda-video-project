CC=gcc
CFLAGS=-c -Wall
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -g
LD_FLAGS    = -lcudart
EXE         = ppm
OBJ         = ppm.o


default: parallel

ppm.o: ppm.cu ppmKernel.cu
	$(NVCC) -c -o $@ ppm.cu $(NVCC_FLAGS)

parallel: $(OBJ)
	$(NVCC) $(OBJ) -o par_3D_conv $(LD_FLAGS)

serial: ppm_3D_serial.c
	$(CC) ppm_3D_serial.c -lm -o ser_3D_conv

clean:
	rm -rf *.o par_3D_conv ser_3D_conv ../outfiles/tmp*
