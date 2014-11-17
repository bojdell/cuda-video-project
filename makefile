CC=gcc
CFLAGS=-c -Wall

all: serial_test

serial_test: serial_test.o
	$(CC) serial_test.o -o serial_test

serial_test.o: serial_test.c
	$(CC) $(CFLAGS) serial_test.c

clean:
	rm -rf *o serial_test