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

multiFrame/ppm2DMult.o: multiFrame/ppm2DMult.cu multiFrame/ppmKernel.cu
	$(NVCC) -c -o multiFrame/ppm2DMult.o multiFrame/ppm2DMult.cu $(LD_FLAGS)

convolution: convolution.o
	$(NVCC) convolution.o -o convolution $(LD_FLAGS)

bandw: bandw.o
	$(NVCC) bandw.o -o bandw $(LD_FLAGS)

serial-convolution: ppm_serial.c
	$(CC) ppm_serial.c -lm -o serial-convolution -D CONV

serial-bw: ppm_serial.c
	$(CC) ppm_serial.c -lm -o serial-bw -D BANDW

multiFrame/ppm2DMult: multiFrame/ppm2DMult.o
	$(NVCC) multiFrame/ppm2DMult.o -o multiFrame/ppm2DMult $(LD_FLAGS)

clean:
	rm -rf *.o bandw convolution serial-bw serial-convolution ppm_serial outfiles/tmp* *.mp4