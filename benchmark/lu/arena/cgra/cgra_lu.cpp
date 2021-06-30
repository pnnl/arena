// =======================================================================
// lu.cpp
// =======================================================================
// ARENA implementation of LU 
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "../../../lib/ARENA.h"

#define SIZE 128
#define NODES 4

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
int __attribute__ ((noinline)) ARENA_kernel(int start, int end, int param) {
  int rank = ARENA_local_rank;
  int N = SIZE;
  int i = start;
//  for(int i=ARENA_local_rank*SIZE/NODES; i<(ARENA_local_rank+1)*SIZE/NODES; ++i) {
//    for(int j=0; j<i; ++j) {
//      for(int k=0; k<j; ++k) {
//        local_A[i][j] -= (local_A[i][k] * local_A[k][j]);
//      }
//      local_A[i][j] /= local_A[j][j];
//    }
//    for(int j=i; j<N; ++j) {
//      for(int k=0; k<i; ++k) {
//        local_A[i][j] -= (local_A[i][k] * local_A[k][j]);
//      }
//    }
//  }
  for(int x = 0; x < N; ++x) {
    int j = x / i;
    int k = x % i;
    if(j < i) {
      if(k < j) {
        local_A[i][j] -= local_A[i][k] * local_A[k][j];
      }
      if(k == i - 1) {
        local_A[i][j] /= local_A[j][j];
      }
    } else {
      if(k < i) {
        local_A[i][j] -= local_A[i][k] * local_A[k][j];
      }
    }
    if(x==N-1)
      spawn(rank+1);
  }

//  cout<<"[result] rank "<<rank<<" local_A: "<<endl;
//  for(int i=0; i<SIZE; ++i)
//    cout<<local_A[rank][i]<<" ";
//  cout<<endl;

//  if(rank<3) {
//    ARENA_spawn[0].id         = ARENA_NORMAL_TASK;
//    ARENA_spawn[0].start      = rank+1;
//    ARENA_spawn[0].end        = rank+2;
//    ARENA_spawn[0].param      = -1;
//    ARENA_spawn[0].more_from  = rank;
//    ARENA_spawn[0].more_start = 0;
//    ARENA_spawn[0].more_end   = SIZE * SIZE/NODES * (rank + 1);
//    // Same as more_start and more_end but need indicate destination (rank+1 for lu)
//    ARENA_remote_ask_start[rank+1] = 0;
//    ARENA_remote_ask_end[rank+1] = SIZE * SIZE/NODES * (rank + 1);
//  }
  return 1;
}


