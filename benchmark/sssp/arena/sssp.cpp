// =======================================================================
// bfs.cpp
// =======================================================================
// ARENA implementation of BFS
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "../../../lib/ARENA.h"
#include <iostream>
#include <fstream>
#include <string>

//#define SIZE 4096
#define SIZE 16
//#define SIZE 16384

#define NODES 4
//#define NODES 16

#define DFS_KERNEL 1

int local_rank;
int local_start;
int local_end;

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
int** adj;
int* path;

void init_data() {
  adj = new int*[SIZE/NODES];
  path = new int[SIZE/NODES];
  for(int i=0; i<SIZE/NODES; ++i) {
    adj[i] = new int[SIZE];
    path[i] = -1;
    for(int j=0; j<SIZE; ++j) {
      adj[i][j] = 0;
    }
  }

  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      if(j == 2*i+1 or j == 2*i+2)
        adj[i-local_start][j] = SIZE;
    }
  }
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params.
// TODO: user specified.
// ----------------------------------------------------------------------
void dfs_kernel(long long int start, long long int end, int param, bool require_data, int length) {
  int level = param;
  int k = 0;
  for(int j=0; j<SIZE; ++j) {
    if(adj[start][j] > level) {
      adj[start][j] = level;
      ARENA_spawn_task(DFS_KERNEL, j, j+1, level+1);
    }
  }
  if(path[start] == -1 or path[start] > level)
    path[start] = level;
}

// ----------------------------------------------------------------------
// Main function.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // initialize global data start and end
  local_rank = ARENA_init(NODES);

  // TODO: declare local data range
  local_start = local_rank * (SIZE / NODES);
  local_end   = (local_rank + 1) * (SIZE / NODES);
  ARENA_set_local(local_start, local_end);

  // TODO: register kernel
  bool isRoot    = true;
  long long int root_start = 1; //single source point
  long long int root_end   = 2;
  int root_param = 1;
  ARENA_register_task(DFS_KERNEL, &dfs_kernel, isRoot, root_start, root_end, root_param);
  // ARENA_register_task(ARENA_NORMAL_TASK, &kernel0);
  // ARENA_register_task(ARENA_NORMAL_TASK, &kernel1);
  // ARENA_register_task(ARENA_NORMAL_TASK, &kernek2);

  // init local data
  init_data();

  // execute ARENA runtime 
  ARENA_run();

  // TODO: Output
  if(local_rank != -1) {
  cout<<"[final] local_rank "<<local_rank<<endl;
  for(int i=local_start; i<local_end; ++i) {
    cout<<path[i-local_start]<<endl;
  }}

  return 0;
}

