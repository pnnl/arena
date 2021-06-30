// =======================================================================
// gemm.h
// =======================================================================
// ARENA tutorial for GEMM (D = alpha x A x B + beta x C).
//
// Author : Cheng Tan
//   Date : March 17, 2021

#include "../../lib/ARENA.h"
#include <iostream>
#include <fstream>
#include <string>

#define SIZE 1024
#define NODES 4

// ----------------------------------------------------------------------
// Global data.
// ----------------------------------------------------------------------
float A[SIZE][SIZE];
float B[SIZE][SIZE];
float C[SIZE][SIZE];
float D[SIZE][SIZE]; //gold output for verification

// ----------------------------------------------------------------------
// Local data allocated onto each node.
// ----------------------------------------------------------------------
float local_A[SIZE/NODES][SIZE];
float local_B_trans[SIZE][SIZE];
float local_C[SIZE/NODES][SIZE];
float local_D[SIZE/NODES][SIZE];
float gold[SIZE][SIZE];
float alpha = 2;
float beta  = 1;

// ----------------------------------------------------------------------
// ARENA local variables. Can be customized by the user.
// ----------------------------------------------------------------------
int local_rank;
int local_start;
int local_end;

// ----------------------------------------------------------------------
// Initialize random local data value for the demo.
// ----------------------------------------------------------------------
void init_data();
bool verify();

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc)
// user specified
// ----------------------------------------------------------------------
void ARENA_kernel(int start, int end, int param, bool require_data, int length) {
  int steps = param;
  int col_start = (local_start - steps*(SIZE/NODES) + SIZE)%SIZE;
  int col_end = col_start+(SIZE/NODES);
  if(require_data) {
    float* addr = *(local_B_trans + col_start);
    for(int i=0; i<length; ++i) {
      *(addr+i) = ARENA_recv_data_buffer[i];
    }
  }
  for(int row=0; row<SIZE/NODES; ++row) {
    int x = 0;
    for(int col = col_start; col<col_end; ++col) {
      for(int k=0; k<SIZE; ++k) {
          // TODO: local_D[row][col] += alpha * local_A[][] * local_B_trans[][];
          local_D[row][col] += alpha * local_A[row][k] * local_B_trans[col][k];
      }
      // TODO: local_D[row][col] += beta * local_C[][];
      local_D[row][col] += local_C[row][col]*beta;
    }
  }
  if(steps < NODES - 1) {
    // TODO: essential data deliver indicated by the spawned task
    int index = (local_start - steps*(SIZE/NODES) + SIZE) % SIZE;
    float* addr = *(local_B_trans + index);
    // TODO: data range that the spawned task should work on
    //ARENA_spawn_task(ARENA_NORMAL_TASK, /*local_end*/, /*local_end+SIZE/NODES*/,
    //                 steps+1, addr, SIZE*SIZE/NODES);
    ARENA_spawn_task(ARENA_NORMAL_TASK, local_end%SIZE, local_end%SIZE+SIZE/NODES,
                     param+1, addr, SIZE*SIZE/NODES);
  }
}

int main(int argc, char *argv[]) {

  // Initialize global data start and end
  local_rank = ARENA_init(NODES);
  local_start = local_rank * (SIZE/NODES);
  local_end   = local_rank * (SIZE/NODES) + (SIZE/NODES);
  ARENA_set_local(local_start, local_end);

  // Register kernel
  bool isRoot = true;
  int root_start = 0;
  int root_end   = SIZE;
  int root_param = 0;
  ARENA_register_task(ARENA_NORMAL_TASK, &ARENA_kernel, isRoot, root_start, root_end, root_param);

  // Initialize local allocated data
  init_data();

  // Execute kernel
  ARENA_run();

  // Verify and Output
  if(!verify()) {
    cout<<"[error] rank "<<local_rank<<endl;
    for(int i=local_start; i<local_end; ++i) {
      for(int j=0; j<SIZE; ++j) {
        cout<<local_D[i-local_start][j]<<" ";
      }
      cout<<endl;
    }
  } else {
    cout<<"[success] rank "<<local_rank<<endl;
  }

  return 0;
}


// ----------------------------------------------------------------------
// helper functions
// ----------------------------------------------------------------------
void display_input() {
  if(local_rank == 0) {
    cout<<"[init A] global "<<endl;
    for(int i=0; i<SIZE; ++i) {
      for(int j=0; j<SIZE; ++j) {
        cout<<A[i][j]<<" ";
      }
      cout<<endl;
    }
    cout<<"[init B] global "<<endl;
    for(int i=0; i<SIZE; ++i) {
      for(int j=0; j<SIZE; ++j) {
        cout<<B[i][j]<<" ";
      }
      cout<<endl;
    }
    cout<<"[init C] global "<<endl;
    for(int i=0; i<SIZE; ++i) {
      for(int j=0; j<SIZE; ++j) {
        cout<<C[i][j]<<" ";
      }
      cout<<endl;
    }
  }

  cout<<"[init A] rank "<<local_rank<<endl;
  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      cout<<local_A[i-local_start][j]<<" ";
    }
    cout<<endl;
  }
  cout<<"[init B] rank "<<local_rank<<endl;
  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      cout<<local_B_trans[i][j]<<" ";
    }
    cout<<endl;
  }
  cout<<"[init C] rank "<<local_rank<<endl;
  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      cout<<local_C[i-local_start][j]<<" ";
    }
    cout<<endl;
  }
  if(local_rank == 0) {
    cout<<"[gold] out"<<endl;
    for(int i=0; i<SIZE; ++i) {
      for(int j=0; j<SIZE; ++j) {
        cout<<gold[i][j]<<" ";
      }
      cout<<endl;
    }
  }
}

void init_data() {
  float tmp = 0;
  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      A[i][j] = tmp;
      B[i][j] = j;
      C[i][j] = i;
      tmp++;
    }
  }

  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      local_A[i-local_start][j] = A[i][j];
      if(local_rank == i/(SIZE/NODES))
        local_B_trans[i][j] = B[j][i];
      else
        local_B_trans[i][j] = -1;
      local_C[i-local_start][j] = C[i][j];
      local_D[i-local_start][j] = 0;
    }
  }

  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      gold[i][j] = 0;
    }
  }

  chrono::system_clock::time_point start = chrono::system_clock::now();
  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      for(int k=0; k<SIZE; ++k) {
        gold[i][j] += alpha*A[i][k]*B[k][j];
      }
      gold[i][j] += beta*C[i][j];
    }
  }
  chrono::system_clock::time_point end = chrono::system_clock::now();
  chrono::duration<double> elapsed_seconds = end - start;
  if(local_rank == 0)
    cout<<"baseline execution time: "<<elapsed_seconds.count()<<"s"<<endl;
}

bool verify() {
  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      if(abs(gold[i][j]-local_D[i-local_start][j])>0.001)
        return false;
    }
  }
  return true;
}


