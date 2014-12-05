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
all: serial_test libav_test

serial_test: serial_test.o
	$(CC) serial_test.o $(LDFLAGS) -o serial_test

serial_test.o: serial_test.c
	$(CC) $(CFLAGS) serial_test.c $(INCLUDES)


libav_test: libav_test.o
	$(CC) libav_test.o $(LDFLAGS) -o libav_test


libav_test.o: libav_test.c
	$(CC) $(CFLAGS) libav_test.c $(INCLUDES)

clean:
	rm -rf *o serial_test