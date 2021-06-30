# C compiler
CC=mpic++
CC_FLAGS=-fPIC -O3 -std=c++11
LDFLAGS=-L$(DUMPI_PATH)/lib
#LDFLAGS=

all: needle

needle: 
	$(CC) $(CC_FLAGS) $(LDFLAGS) nw.cpp -o needle

clean:
	rm -f needle
