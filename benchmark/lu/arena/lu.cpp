// =======================================================================
// lu.cpp
// =======================================================================
// ARENA implementation of LU 
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "../../../lib/ARENA.h"

#define SIZE 1024
#define NODES 4

int local_rank;
int local_start;
int local_end;

// ----------------------------------------------------------------------
// Total data allocated onto nodes.
// TODO: user specified.
// ----------------------------------------------------------------------
float A[SIZE][SIZE];// = {{0,1,2,3}, {1,2,3,4}, {2,3,4,5}, {3,4,5,6}};
float B[SIZE][SIZE];// = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
float local_A[SIZE][SIZE];// = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};

void init_local_data() {
  int N = SIZE;
  int rank = local_rank;
  for(int i=0; i<N; ++i) {
    for(int j=0; j<i+1; ++j)
      A[i][j] = -(j % N) / (N*1.0) + 1.0;
    for(int j=i+1; j<N; ++j)
      A[i][j] = 0.0;
    A[i][i] = 1.0;
  }
  for(int t=0; t<N; ++t)
    for(int r=0; r<N; ++r)
      for(int s=0; s<N; ++s)
        B[r][s] += A[r][t] * A[s][t];
  for(int r=0; r<N; ++r)
    for(int s=0; s<N; ++s)
      A[r][s] = B[r][s];
  for(int i=local_start*SIZE/NODES; i<local_end*SIZE/NODES; ++i)
    for(int j=0; j<SIZE; ++j)
      local_A[i][j] = A[i][j];
//  cout<<"[init] rank "<<rank<<" local_A: "<<endl;
//  for(int i=0; i<SIZE; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      cout<<local_A[i][j]<<" ";
//    }
//    cout<<endl;
//  }
  // cout<<"[init] rank "<<rank<<" local_A: ";
  // for(int i=0; i<SIZE; ++i)
  //   cout<<local_A[rank][i]<<" ";
  // cout<<endl;
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// TODO: user specified.
// ----------------------------------------------------------------------
void ARENA_kernel(long long int start, long long int end, int param, bool require_data, int length) {
  int rank = local_rank;
  int N = SIZE;

  if(require_data) {
    for(int i=0; i<length; ++i) {
      local_A[i/SIZE][i%SIZE] = ARENA_recv_data_buffer[i];
    }
  }

  for(int i=rank*SIZE/NODES; i<(rank+1)*SIZE/NODES; ++i) {
    for(int j=0; j<i; ++j) {
      for(int k=0; k<j; ++k) {
        local_A[i][j] -= (local_A[i][k] * local_A[k][j]);
      }
      local_A[i][j] /= local_A[j][j];
    }
    for(int j=i; j<N; ++j) {
      for(int k=0; k<i; ++k) {
        local_A[i][j] -= (local_A[i][k] * local_A[k][j]);
      }
    }
  }
  if(rank<NODES-1) {
    ARENA_spawn_task(ARENA_NORMAL_TASK, rank+1, rank+2, -1, local_A[0], SIZE * SIZE/NODES * (rank + 1));
  }
}

// ----------------------------------------------------------------------
// Main function. No need to change.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // Initialize global data start and end
  local_rank = ARENA_init(NODES);

  // Initialize local allocated data
//  init_kernel();
  init_local_data();

  local_start = local_rank;// * (num_vertice/NODES);
  local_end   = local_rank+1;// * (num_vertice/NODES) + (num_vertice/NODES);
  ARENA_set_local(local_start, local_end);

  // Register kernel
  long long int root_start = 0;
  long long int root_end = 1;
  int root_param = 0;
  ARENA_register_task(ARENA_NORMAL_TASK, &ARENA_kernel, true, root_start, root_end, root_param);


  // Execute kernel
  ARENA_run();

  // Output
//  if(ARENA_local_rank == NODES - 1) {
//    cout<<"[output] rank "<<ARENA_local_rank<<endl;
//    for(int i=0; i<SIZE; ++i) {
//      for(int j=0; j<SIZE; ++j) {
//        cout<<local_A[i][j]<<" ";
//      }
//      cout<<endl;
//    }
//  }

  return 0;
}

/*
// ----------------------------------------------------------------------
// Prepare data to send to remote nodes.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
void ARENA_load_data(int start, int end, float* buff) {
  for(int i=start; i<end; ++i) {
    buff[i] = local_A[i/SIZE][i%SIZE];
  }
  // cout<<"[print] prepare data: "<<buff[0]<<" "<<buff[1]<<" "<<buff[2]<<" "<<buff[3]<<endl;
}

// ----------------------------------------------------------------------
// Receive data from remote nodes and store into local memory.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
void ARENA_store_data(int start, int end, int source, float* buff) {
  // cout<<"[print] received data: "<<buff[0]<<" "<<buff[1]<<" "<<buff[2]<<" "<<buff[3]<<endl;
  for(int i=start; i<end; ++i) {
    local_A[i/SIZE][i%SIZE] = buff[i];
  }
  // cout<<"[print] source "<<source<<" new local_A: "<<endl;
  // for(int i=0; i<SIZE; ++i) {
  //   for(int j=0; j<SIZE; ++j) {
  //     cout<<local_A[i][j]<<" ";
  //   }
  //   cout<<endl;
  // }
}
*/
