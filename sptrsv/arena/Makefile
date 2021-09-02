CXX=mpic++ -std=c++11 -O3 

HIPARTI_PATH=/qfs/people/xiec066/HiParTI

#INC=-I/qfs/people/xiec066/HiParTI/include
INC=-I$(HIPARTI_PATH)/include
LIB=-L$(HIPARTI_PATH)/build -lHiParTI -lm


all:
	$(CXX) $(INC)  -o sptrsv_hicoo sptrsv_hicoo.cpp $(LIB)
#$(CXX) $(INC) -o sptrsv_hicoo sptrsv_hicoo.o $(LIB)
clean:
	-rm sptrsv_hicoo
