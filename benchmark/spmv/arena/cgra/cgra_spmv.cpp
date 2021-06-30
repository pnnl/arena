// =======================================================================
// spmv.h
// =======================================================================
// ARENA implementation of SPMV.
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "../../../lib/ARENA.h"

#define TOTAL 80
#define SIZE TOTAL*TOTAL/4 // Assume 25% sparsity.
#define NODES 4 // define number of nodes

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

int spawn_index = 0;
void __attribute__ ((noinline)) spawn(int j){
  ARENA_spawn[spawn_index].id    = ARENA_NORMAL_TASK;
  ARENA_spawn[spawn_index].start = j;
  ARENA_spawn[spawn_index].end   = j+1;
//  ARENA_spawn[k].param = level+1;
  spawn_index++;
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// TODO: user specified.
// ----------------------------------------------------------------------
int kernel_times = 0;
int __attribute__ ((noinline)) ARENA_kernel(int start, int end, int param) {
  bool never = true;
  int result_index = 0;
//  #pragma clang loop unroll_count(8)
//  for(int x=0; x<(TOTAL*SIZE); ++x) {
//    int i=x/TOTAL;
//    int j=1+x%TOTAL;
//    if(i+ARENA_local_bound < global_ROW[j] and never) {
//      result_index = j-1;
//      never = false;
//    }
//    if(j==TOTAL) {
//      never = true;
//      local_OUT[result_index] += local_V[i] * local_X[local_COL[i]];
//    }
//  }

  #pragma clang loop unroll_count(32)
  for(int i=0; i<end-start; ++i) {
    int result_index = 0;
    local_OUT[0] += local_V[i] * local_X[local_COL[i]];
  }
  return 0;
}


