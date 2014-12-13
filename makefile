CC=gcc
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart
OBJ	        = ppm.o


default: parallel

ppm.o: ppm.cu ppmKernel.cu
	$(NVCC) -c -o $@ ppm.cu $(NVCC_FLAGS)

parallel: $(OBJ)
	$(NVCC) $(OBJ) -o ppm $(LD_FLAGS)

serial: ppm_serial.c
	$(CC) ppm_serial.c -lm -o ppm_serial

clean:
	rm -rf *.o ppm ppm_serial