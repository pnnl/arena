# C compiler
CC = mpic++
ICC = mpicxx
CC_FLAGS = -fPIC -O3
OFFLOAD_CC_FLAGS = -offload-option,mic,compiler,"-no-opt-prefetch"

all: needle needle_offload

needle: 
	$(CC) $(CC_FLAGS) needle.cpp -o needle 

needle_offload:
	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD needle.cpp -o needle_offload

clean:
	rm -f needle needle_offload
