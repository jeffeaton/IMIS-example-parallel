CC= gcc-4.9.1 # gcc
CBLAS= -lgslcblas  
# CFLAGS= -lgsl -O2 -std=c99 
# EXEC = trimodal-single
CFLAGS= -lgsl -O2 -std=c99 -fopenmp
EXEC = trimodal-parallel

all: 
	$(CC) $(CFLAGS) $(CBLAS) main.c imis.c likelihood.c -o $(EXEC)
