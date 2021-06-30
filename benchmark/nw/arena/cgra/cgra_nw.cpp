//================================
//nw.cpp
//================================
//ARENA implementation of NW
//
//


#include "../../../lib/ARENA.h"
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define NUM_THREAD 4
#define SIZE 4
#define NODES 4
#define BLOCK_SIZE 32
#define ROW 128
#define PENALTY 10

    //inital local data
	 
	int reference_blk [SIZE][SIZE][(BLOCK_SIZE) *(BLOCK_SIZE)]; 
	int input_itemsets_blk [SIZE][SIZE][(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)]; 
        int node_param;
 
inline int maximum( int a,
                 int b,
                 int c){

        int k;
        if( a <= b )
                k = b;
        else
        k = a;

        if( k <=c )
        return(c);
        else
        return(k);
}

int spawn_index = 0;
void __attribute__ ((noinline)) spawn(int j){
  ARENA_spawn[spawn_index].id    = ARENA_NORMAL_TASK;
  ARENA_spawn[spawn_index].start = j;
  ARENA_spawn[spawn_index].end   = j+1;
//  ARENA_spawn[k].param = level+1;
  spawn_index++;
}

int __attribute__ ((noinline)) ARENA_kernel(int start, int end, int param) {

  int rank = ARENA_local_rank;
  //if(rank ==0) param = node_param;
  int x_index = rank;
  int y_index = param;

  int i=1;
  for ( int j = 1; j < BLOCK_SIZE + 1; ++j)
  {
    input_itemsets_blk[x_index][y_index][i*(BLOCK_SIZE + 1) + j] = maximum( input_itemsets_blk[x_index][y_index][(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_blk[x_index][y_index][(i - 1)*BLOCK_SIZE + j - 1],
    input_itemsets_blk[x_index][y_index][i*(BLOCK_SIZE + 1) + j - 1] - PENALTY,
    input_itemsets_blk[x_index][y_index][(i - 1)*(BLOCK_SIZE + 1) + j] - PENALTY);
    if(j==BLOCK_SIZE) {
      if ((x_index == 0)&&(y_index<3)){   
        spawn(rank);
      }
      if(x_index < 3) {
        spawn(rank+1);
      }
    }
  }
  node_param++;   
         
return -1;

}

