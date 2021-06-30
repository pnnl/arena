// =======================================================================
// spmv.h
// =======================================================================
// ARENA implementation of SPMV.
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "../../../lib/ARENA.h"

#define SIZE 1024
#define NODES 8

// ----------------------------------------------------------------------
// Total data allocated onto nodes TODO: user specified 
// ----------------------------------------------------------------------
int A[SIZE][SIZE];// = {{1,2,3,4}, {2,3,4,5}, {3,4,5,6}, {4,5,6,7}};
int B[SIZE][SIZE];// = {{1,2,1,2}, {1,2,1,2}, {1,2,1,2}, {1,2,1,2}};
int C[SIZE][SIZE];// = {{1,2,1,2}, {1,2,1,2}, {1,2,1,2}, {1,2,1,2}};

// ----------------------------------------------------------------------
// Local data allocated onto a node TODO: user specified
// ----------------------------------------------------------------------
int local_A[SIZE/NODES][SIZE];// = {{0,0,0,0}};
int local_B[SIZE][SIZE];
int local_C[SIZE/NODES][SIZE];
int local_OUT[SIZE/NODES][SIZE];// = {{0,0,0,0}};
int alpha = 2;
int beta  = 1;

int spawn_index = 0;
void __attribute__ ((noinline)) spawn(int j){
  ARENA_spawn[spawn_index].id    = ARENA_NORMAL_TASK;
  ARENA_spawn[spawn_index].start = j;
  ARENA_spawn[spawn_index].end   = j+1;
//  ARENA_spawn[k].param = level+1;
  spawn_index++;
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc) TODO: user specified
// ----------------------------------------------------------------------
int total_times = 0;
int __attribute__((noinline)) ARENA_kernel(int start, int end, int param) {
  int col_start = (ARENA_local_start - total_times*(SIZE/NODES) + SIZE)%SIZE;
  int col_end = col_start+(SIZE/NODES);
  #pragma clang loop unroll_count(8)
  for(int x = col_start*SIZE/NODES*SIZE; x<col_end*SIZE/NODES*SIZE; ++x) {
    int row = (x-col_start)/SIZE/(col_end-col_start);
    int col = (x-col_start)/SIZE%(col_end-col_start) + col_start;
    int i = (x-col_start)%SIZE;
    local_OUT[row][col] += alpha * local_A[row][i] * local_B[col][i];
    if(i == SIZE-1)
      local_OUT[row][col] += local_C[row][col]*beta;
    if(x == col_end*SIZE/NODES*SIZE-1)
      spawn(end);
  }
//  for(int row=0; row<SIZE/NODES; ++row) {
//    for(int col = col_start; col<col_end; ++col) {
//      for(int i=0; i<SIZE; ++i) {
//        local_OUT[row][col] += alpha * local_A[row][i] * local_B[col][i];
//      }
//      local_OUT[row][col] += local_C[row][col]*beta;
//    }
//  }
  total_times += 1;
  return total_times;
}

