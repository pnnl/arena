//================================
//nw.cpp
//================================
//ARENA implementation of NW
//
//


#include "../../lib/ARENA.h"
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define SIZE 16
#define NODES SIZE
#define NUM_THREAD NODES
#define ROW 8092
#define BLOCK_SIZE ROW/NUM_THREAD
#define PENALTY 10

    //inital local data
	 
	int reference_blk [SIZE][SIZE][(BLOCK_SIZE+1) *(BLOCK_SIZE+1)]; 
	int input_itemsets_blk [SIZE][SIZE][(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)]; 
        int node_param;
 
int maximum( int a,
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

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

//rows = 128
//penalty = 10
//omp_num_threads = 4 

void init_kernel(int rows, int penalty ) 
{
    int max_rows, max_cols;
   
    node_param = 0;
    
    max_rows = rows;
    max_cols = rows;
    
    

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    
    //allocate global data
    int * referrence = new int [max_rows * max_cols];
    int * input_itemsets = new int [max_rows * max_cols];

    if (!input_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );

    for (int i = 0 ; i < max_cols; i++){
        for (int j = 0 ; j < max_rows; j++){
            input_itemsets[i*max_cols+j] = 0;
        }
    }
    
    
   if(ARENA_local_rank == 0)
    printf("Start Needleman-Wunsch\n");
    
    //define globle matrix

    for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
        input_itemsets[i*max_cols] = rand() % 10 + 1;
    }
    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
        input_itemsets[j] = rand() % 10 + 1;
    }


    for (int i = 1 ; i < max_cols; i++){
        for (int j = 1 ; j < max_rows; j++){
            referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
        }
    }

    for( int i = 1; i< max_rows ; i++)
        input_itemsets[i*max_cols] = -i * penalty;
    for( int j = 1; j< max_cols ; j++)
        input_itemsets[j] = -j * penalty;

  
  
  int totalBlock = SIZE;
  
   if(ARENA_local_rank == 0)
  printf("inital local data\n");

	for( int blk_x = 0; blk_x < totalBlock; blk_x++ )
    {
		for( int blk_y = 0; blk_y < totalBlock; blk_y++ )
		{
			// Copy referrence to block memory
            for ( int i = 0; i < BLOCK_SIZE; ++i )
			{
                for ( int j = 0; j < BLOCK_SIZE; ++j)
                {
                    reference_blk[blk_x][blk_y][i*BLOCK_SIZE + j] = referrence[max_cols*(blk_y*BLOCK_SIZE + i + 1) + blk_x*BLOCK_SIZE +  j + 1];
                }
            }

			// Copy input to block memory
			for ( int i = 0; i < BLOCK_SIZE+1; ++i )
			{
                for ( int j = 0; j < BLOCK_SIZE+1; ++j)
                {
					input_itemsets_blk[blk_x][blk_y][i*(BLOCK_SIZE + 1) + j] = input_itemsets[max_cols*(blk_y*BLOCK_SIZE + i) + blk_x*BLOCK_SIZE +  j];
                }
            }
		}
	}

    //kernel init finish
     if(ARENA_local_rank == 0){
    printf("Num of threads: %d\n", NUM_THREAD);
    printf("Num of blk: %d\n", totalBlock*totalBlock);  
    }
    //init finish
    
}    
    
                   



int ARENA_kernel(int start, int end, int param) {

int rank = ARENA_local_rank;
//if(rank ==0) param = node_param;
int spawn = 0;
int x_index = rank;
int y_index = param;


//TODO: move index for next tasks
   
   //   printf("calculating in task x: %d y: %d\n", x_index, y_index);
            //Receive input from outside

            // Compute  // mpi compute
            for ( int i = 1; i < BLOCK_SIZE + 1; ++i )
            {
                for ( int j = 1; j < BLOCK_SIZE + 1; ++j)
                {
                    input_itemsets_blk[x_index][y_index][i*(BLOCK_SIZE + 1) + j] = maximum( input_itemsets_blk[x_index][y_index][(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_blk[x_index][y_index][(i - 1)*BLOCK_SIZE + j - 1],
                            input_itemsets_blk[x_index][y_index][i*(BLOCK_SIZE + 1) + j - 1] - PENALTY,
                            input_itemsets_blk[x_index][y_index][(i - 1)*(BLOCK_SIZE + 1) + j] - PENALTY);
                }
            }
           
			
			
			//copy result to spawn >
			//if(x_index != (max_cols-1)/BLOCK_SIZE){
		//	for ( int i = 0; i < BLOCK_SIZE+1; ++i)
         //   {
		//		input_itemsets_blk[x_index+1][y_index][i*(BLOCK_SIZE + 1)] = input_itemsets_blk[x_index][y_index][(i)*(BLOCK_SIZE + 1) + BLOCK_SIZE + 1];
         //   }
			//}

			//copy result to spam V
			if(y_index != (SIZE-1)){
				for ( int j = 0; j < BLOCK_SIZE+1; ++j)
				{
					input_itemsets_blk[x_index][y_index+1][j] = input_itemsets_blk[x_index][y_index][(BLOCK_SIZE + 1)*(BLOCK_SIZE + 1) + j];
				}
			}
      
#ifdef DEBUG
      if(x_index == 1)
      printf("calculated in task x: %d y: %d\n", x_index, y_index);
#endif      
      
       //generate spawn V
			if ((x_index == 0)&&(y_index<(SIZE-1))){  
#ifdef DEBUG 
      if(x_index == 1)
      printf("generate V\n");
#endif         
      ARENA_spawn[spawn].id         = ARENA_NORMAL_TASK;
      ARENA_spawn[spawn].start      = rank;
      ARENA_spawn[spawn].end        = rank+1;
      ARENA_spawn[spawn].param      = param+1;	
      spawn++;	  
			}
      
      //generate spawn >
			if(x_index < (SIZE-1)) {
#ifdef DEBUG
       printf("generate >\n");
#endif
      ARENA_spawn[spawn].id         = ARENA_NORMAL_TASK;
      ARENA_spawn[spawn].start      = rank+1;
      ARENA_spawn[spawn].end        = rank+2;
      ARENA_spawn[spawn].param      = param; //param for spawn
      ARENA_spawn[spawn].more_from  = rank;
      ARENA_spawn[spawn].more_start = 0;
      ARENA_spawn[spawn].more_end   = (BLOCK_SIZE+1);
    // Same as more_start and more_end but need indicate destination (rank+1 for nw when using right pass)
      ARENA_remote_ask_start[rank+1] = 0;
      ARENA_remote_ask_end[rank+1] = (BLOCK_SIZE+1);
      spawn++;
      }
      
      node_param++;   
         
      if(x_index == (SIZE-1)) {
     printf("line %d finished\n\n", y_index);
     }
     
      
return spawn;

}

// ----------------------------------------------------------------------
// Initialize task start point, data tag, and remote data requirement.
// TODO: user specified
// ----------------------------------------------------------------------
void ARENA_init_task(int argc, char *argv[], int nodes) {
  // MPI initial
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  ARENA_nodes = nodes;
  ARENA_local_rank = rank;
  
  
  // TODO: Task start point.
  ARENA_global_start = 0;
  ARENA_global_end   = 1;

  // TODO: Data tag.
  ARENA_local_bound = rank;
  ARENA_local_start  = rank;
  ARENA_local_end    = rank + 1;

  // TODO: Remote data requirement.
  ARENA_init_data_buff(BLOCK_SIZE+1,true);
  
  if(rank>0)
  ARENA_local_need_buff[rank-1] = new float[(BLOCK_SIZE+1)]; 
  
 
  if(rank<(SIZE-1)) 
  ARENA_remote_ask_buff[rank+1] = new float[(BLOCK_SIZE+1)];
    
}

    
// ----------------------------------------------------------------------
// // Main function. No need to change.
// // ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

    // Initialize global data start and end
    ARENA_init_task(argc, argv, NODES);

    // Register kernel
    ARENA_register(ARENA_NORMAL_TASK, &ARENA_kernel, true);

    // Initialize local allocated data
    init_kernel(ROW,PENALTY) ;

    // Execute kernel
    ARENA_run();
    
    return 0;
}


// ----------------------------------------------------------------------
// Prepare data to send to remote nodes.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
void ARENA_load_data(int start, int end, float* buff) {

  int x_index = ARENA_local_rank;
  int y_index = node_param-1;
  
  //send to right
  if(x_index != (SIZE-1)){
	 for(int i=start; i<end; ++i) {
		buff[i] = input_itemsets_blk[x_index][y_index][(i)*(BLOCK_SIZE + 1) + BLOCK_SIZE + 1];
	 }
  }
#ifdef DEBUG
  printf("load to right\n\n");
#endif 
}

// ----------------------------------------------------------------------
// Receive data from remote nodes and store into local memory.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
void ARENA_store_data(int start, int end, int source, float* buff) {

  int x_index = ARENA_local_rank;
  int y_index = node_param;
   
  //receive from left
  if(x_index != 0){
	for(int i=start; i<end; ++i) {
		input_itemsets_blk[x_index][y_index][(i)*(BLOCK_SIZE + 1) + BLOCK_SIZE + 1] = buff[i];
	 }
  }
#ifdef DEBUG  
   printf("receive from left\n");
#endif  
}
