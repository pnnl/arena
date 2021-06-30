//  =======================================================================
//  nw.cpp
//  =======================================================================
//  ARENA template for implementation
//

#include "./ARENA.h"
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define SIZE 16
#define NODES SIZE
                   
int ARENA_kernel(int start, int end, int param) {
  // TODO: 

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
   
}
    
// ----------------------------------------------------------------------
// Main function. No need to change.
// ----------------------------------------------------------------------
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

}

// ----------------------------------------------------------------------
// Receive data from remote nodes and store into local memory.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
void ARENA_store_data(int start, int end, int source, float* buff) {

}
