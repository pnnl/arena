// =======================================================================
// spmv.h
// =======================================================================
// ARENA implementation of SPMV.
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "../../../lib/ARENA.h"

#define TOTAL 4000
#define SIZE TOTAL*TOTAL/4 // Assume 25% sparsity.
#define NODES 4 // define number of nodes

int local_rank;
int local_start;
int local_end;

// ----------------------------------------------------------------------
// Total data allocated onto nodes.
// TODO: user specified.
// ----------------------------------------------------------------------
int global_A[TOTAL][TOTAL]; // = {{0,0,0,0}, {5,8,0,0}, {0,0,3,0}, {0,6,0,0}};
int global_V[SIZE]; // = {5,8,3,6};
int global_COL[SIZE]; // = {0,1,2,1};
int global_ROW[TOTAL+1]; // = {0,0,2,3,4};
int global_result[TOTAL];

//int A[4][4] = {{0,0,0,0}, {5,8,0,0}, {0,0,3,0}, {0,6,0,0}};
//int v[4] = {5,8,3,6};
//int col_index[4] = {0,1,2,1};
//int row_index[5] = {0,0,2,3,4};
//int x[4] = {1,2,3,4};

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
int local_V[SIZE/NODES]; // = {0};
int local_COL[SIZE/NODES]; // = {0};
int local_X[TOTAL]; // = {1,2,3,4};
int local_OUT[TOTAL];
int local_RESULT[TOTAL];

//int V[1] = {0};
//int COL_INDEX[1] = {0};
//int X[4] = {0,0,0,0};

void init_data() {
  for(int i=0; i<TOTAL; ++i) {
    for(int j=0; j<TOTAL; ++j) {
      if((i+j)%4 == 0)
        global_A[i][j] = i+j+1;
      else
        global_A[i][j] = 0;
    }
  }
  int k=0;
  global_ROW[0] = 0;
  for(int i=0; i<TOTAL; ++i) {
    for(int j=0; j<TOTAL; ++j) {
      if(global_A[i][j] != 0) {
        global_V[k] = global_A[i][j];
        global_COL[k] = j;
        k++;
      }
    }
    global_ROW[i+1] = k;
  }
  for(int i=0; i<TOTAL; ++i) {
    local_X[i] = i + 1;
    global_result[i] = 0;
  }
//  cout<<"global_ROW: "<<endl;
//  for(int i=0; i<TOTAL; ++i) {
//    cout<<" "<<global_V[i];
//  }
//  cout<<endl;


//  int rank = ARENA_local_rank;
//  V[0] = v[rank];
//  COL_INDEX[0] = col_index[rank];
//  for(int i=0; i<4; ++i) {
//    X[i] = x[i];
//  }

  for(int i=0; i<SIZE/NODES; ++i) {
    local_V[i] = global_V[ARENA_local_rank*SIZE/NODES+i];
    local_COL[i] = global_COL[ARENA_local_rank*SIZE/NODES+i];
  }
  for(int i=0; i<TOTAL; ++i) {
    local_OUT[i] = 0;
  }
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// TODO: user specified.
// ----------------------------------------------------------------------
int kernel_times = 0;
void ARENA_kernel(long long int start, long long int end, int param, bool require_data, int length) {

  if(require_data) {
    for(int i=0; i<length; ++i) {
      local_OUT[i] += ARENA_recv_data_buffer[i];
    }
  }

  if(kernel_times == 0) {
    for(int i=0; i<end-start; ++i) {
      int result_index = 0;
      for(int j=1; j<TOTAL+1; ++j) {
        if(i+local_start < global_ROW[j]) {
          result_index = j-1;
          break;
        }
      }
      local_OUT[result_index] += local_V[i] * local_X[local_COL[i]];
    }
    kernel_times++;
    if(local_rank == 0) {
      ARENA_spawn_task(ARENA_NORMAL_TASK, local_end, local_end+1, 0, (float*)local_OUT, TOTAL);
    }
  } else {
    if(local_rank > 0 and local_rank < NODES - 1) {
      ARENA_spawn_task(ARENA_NORMAL_TASK, local_end, local_end+1, 0, (float*)local_OUT, TOTAL);
    }
  }
}

int main(int argc, char *argv[]) {

  // Initialize global data start and end
  local_rank = ARENA_init(NODES);

  init_data();
  local_start = ARENA_local_rank * (SIZE/NODES);
  local_end   = ARENA_local_rank * (SIZE/NODES) + SIZE/NODES;
  ARENA_set_local(local_start, local_end);

  // Register kernel
  long long int root_start = 0;
  long long int root_end   = SIZE;
  int root_param = 0;
  ARENA_register_task(ARENA_NORMAL_TASK, &ARENA_kernel, true, root_start, root_end, root_param);

  // Execute kernel
  ARENA_run();

  // Output
//  if(ARENA_local_rank == ARENA_nodes - 1) {
//    cout<<"[final] rank "<<ARENA_local_rank<<endl;
//    for(int i=0; i<TOTAL; ++i) {
//      cout<<local_OUT[i]<<" ";
//    }
//    cout<<endl;
//  }

  return 0;
}
