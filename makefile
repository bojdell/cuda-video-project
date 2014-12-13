CC=gcc
LDFLAGS:=-lm

NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart


default: convolution

convolution.o: ppm.cu ppmKernel.cu
	$(NVCC) -c -o convolution.o ppm.cu $(NVCC_FLAGS) -D CONV

bandw.o: ppm.cu ppmKernel.cu
	$(NVCC) -c -o bandw.o ppm.cu $(NVCC_FLAGS) -D BANDW

convolution: convolution.o
	$(NVCC) convolution.o -o convolution $(LD_FLAGS)

bandw: bandw.o
	$(NVCC) bandw.o -o bandw $(LD_FLAGS)

serial: ppm_serial.c
	$(CC) ppm_serial.c -lm -o ppm_serial

clean:
	rm -rf *.o bandw convolution ppm_serial