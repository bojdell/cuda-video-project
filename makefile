# CC=gcc

# # define library paths in addition to /usr/lib
# # LDFLAGS=-L/lib/libavcodec -L/lib/libavutil -l/lib/avcodec

# CFLAGS=-Wall -o $@ -I. -I./lib/libavcodec -I./lib/

# DEPS = serial_test.h 
# OBJ = serial_test.o

# all: serial_test

# %.o: %.c $(DEPS)
# 	$(CC) $(CFLAGS) -c $<

# serial_test: $(OBJ)
# 	$(CC) $(CFLAGS) $^

# .PHONY: clean

# clean:
# 	rm -rf *o serial_test

CC=gcc
CFLAGS=-c -Wall -I/usr/local/include/libavcodec -I/usr/local/include/libavformat
INCLUDES:=$(shell pkg-config --cflags libavformat libavcodec libswscale libavutil libavfilter)
LDFLAGS:=$(shell pkg-config --libs libavformat libavcodec libswscale libavutil libavfilter) -lm

NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart 
EXE	        = ppm
OBJ	        = ppm.o


default: $(EXE)


# ppmTmp: ppmTmp.o
# 	$(CC) ppmTmp.o $(LDFLAGS) -o ppmTmp

# ppmTmp.o: ppmTmp.c
# 	$(CC) $(CFLAGS) ppmTmp.c $(INCLUDES)


ppm.o: ppm.cu ppmKernel.cu 
	$(NVCC) -c -o $@ ppm.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)