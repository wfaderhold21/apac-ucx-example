CC=mpicc
CFLAGS=-O2
LDFLAGS=-lucp -luct -lucs -lucm 


.PHONY: all clean
all: wireup

wireup: wireup.c mpi.c 

clean:
	rm wireup
