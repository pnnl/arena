// =======================================================================
// bfs.cpp
// =======================================================================
// ARENA implementation of BFS
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "../../../lib/ARENA.h"
#include <chrono>
#include <ctime>

#define SIZE 8000
//#define SIZE 4
//#define SIZE 8
#define NODES 4
// ----------------------------------------------------------------------
// Total data allocated onto nodes.
// TODO: user specified.
// ----------------------------------------------------------------------
//int GRAPH[SIZE][SIZE] = {{0,0,0,SIZE}, {SIZE,0,SIZE,0}, {0,0,0,0}, {0,SIZE,0,0}};
//int GRAPH[SIZE][SIZE] = {{0,SIZE,SIZE,SIZE}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}};
int GRAPH[SIZE][SIZE];// = {{0,SIZE,0,0}, {0,0,SIZE,0}, {0,0,0,SIZE}, {0,0,0,0}};
//int GRAPH[SIZE][SIZE] = {{0,0,0,SIZE,0,0,0,0},    {SIZE,0,SIZE,0,SIZE,SIZE,0,0},
//                         {0,0,0,SIZE,0,0,SIZE,0}, {0,SIZE,0,0,0,0,0,0},
//                         {0,0,0,0,0,0,0,0},       {0,0,0,0,0,0,0,SIZE},
//                         {0,0,0,0,0,0,0,SIZE},    {0,0,0,0,0,0,0,0}             };

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
int M[SIZE/NODES][SIZE+1];// = {{0,0,0,0,SIZE}};
//int M[SIZE/NODES][SIZE+1] = {{0,0,0,0,0,0,0,0,SIZE}, {0,0,0,0,0,0,0,0,SIZE}};
// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// TODO: user specified.
// ----------------------------------------------------------------------
int spawn_index = 0;
void __attribute__ ((noinline)) spawn(int j){
  ARENA_spawn[spawn_index].id    = ARENA_NORMAL_TASK;
  ARENA_spawn[spawn_index].start = j;
  ARENA_spawn[spawn_index].end   = j+1;
//  ARENA_spawn[k].param = level+1;
  spawn_index++;
}

int __attribute__ ((noinline)) ARENA_kernel(int start, int end, int param) {
  int level = param;
  spawn_index = 0;
  #pragma clang loop unroll_count(16)
  for(int j=0; j<SIZE; j+=1) {
    if(M[start][j] > level) {
      M[start][j] = level;
      spawn(j);
    }
  }
//  if(M[start][SIZE] > level)
//    M[start][SIZE] = level;
//  if(k > 0) return level + 1;
  return level+1;
}


