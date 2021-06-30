// =======================================================================
// spmv.h
// =======================================================================
// ARENA implementation of SPMV.
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "../../../lib/ARENA.h"
#include <iostream>
#include <fstream>
#include <string>

//#define SIZE 8
#define SIZE 1024
//#define NODES 2
#define NODES 4
//#define NODES 8
//#define NODES 16

// ----------------------------------------------------------------------
// Total data allocated onto nodes TODO: user specified 
// ----------------------------------------------------------------------
float A[SIZE][SIZE];
float B[SIZE][SIZE];
float C[SIZE][SIZE];

// ----------------------------------------------------------------------
// Local data allocated onto a node TODO: user specified
// ----------------------------------------------------------------------
float local_A[SIZE/NODES][SIZE];// = {{0,0,0,0}};
float local_B[SIZE][SIZE];
float local_C[SIZE/NODES][SIZE];
float local_OUT[SIZE/NODES][SIZE];// = {{0,0,0,0}};
float** gold;
float alpha = 2;
float beta  = 1;
int I = SIZE;
int J = SIZE;
int K = SIZE;

int local_rank;
int local_start;
int local_end;

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
        local_B[i][j] = B[j][i];
      else
        local_B[i][j] = -1;
      local_C[i-local_start][j] = C[i][j];
      local_OUT[i-local_start][j] = 0;
    }
  }

  gold = new float*[SIZE];
  for(int i=0; i<SIZE; ++i) {
    gold[i] = new float[SIZE];
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
      if(abs(gold[i][j]-local_OUT[i-local_start][j])>0.001)
        return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc) TODO: user specified
// ----------------------------------------------------------------------
void ARENA_kernel(long long int start, long long int end, int param, bool require_data, int length) {
  int steps = param;
  int col_start = (local_start - steps*(SIZE/NODES) + SIZE)%SIZE;
  int col_end = col_start+(SIZE/NODES);
  if(require_data) {
    float* addr = *(local_B + col_start);
    for(int i=0; i<length; ++i) {
      *(addr+i) = ARENA_recv_data_buffer[i];
    }
  }
  for(int row=0; row<SIZE/NODES; ++row) {
    int x = 0;
    for(int col = col_start; col<col_end; ++col) {
      for(int i=0; i<SIZE; ++i) {
          local_OUT[row][col] += alpha * local_A[row][i] * local_B[col][i];
      }
      local_OUT[row][col] += local_C[row][col]*beta;
    }
  }
  if(steps < NODES - 1) {
    int index = (local_start - steps*(SIZE/NODES) + SIZE) % SIZE;
    float* addr = *(local_B + index);
    ARENA_spawn_task(ARENA_NORMAL_TASK, local_end%SIZE, local_end%SIZE+SIZE/NODES,
                     param+1, addr, SIZE*SIZE/NODES);
  }
}

int main(int argc, char *argv[]) {

  local_rank = ARENA_init(NODES);

  local_start = local_rank * (SIZE/NODES);
  local_end   = local_rank * (SIZE/NODES) + (SIZE/NODES);
  ARENA_set_local(local_start, local_end);

  // Register kernel
  bool isRoot = true;
  long long int root_start = 0;
  long long int root_end   = SIZE;
  int root_param = 0;
  ARENA_register_task(ARENA_NORMAL_TASK, &ARENA_kernel, isRoot, root_start, root_end, root_param);

  init_data();

  // Execute kernel
  ARENA_run();

  // Verify and Output
  if(!verify()) {
    cout<<"[error] rank "<<local_rank<<endl;
    for(int i=local_start; i<local_end; ++i) {
      for(int j=0; j<SIZE; ++j) {
        cout<<local_OUT[i-local_start][j]<<" ";
      }
      cout<<endl;
    }
  } else {
    cout<<"[success] rank "<<local_rank<<endl;
  }

  return 0;
}

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
      cout<<local_B[i][j]<<" ";
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
