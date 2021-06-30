// =======================================================================
// sssp.cpp
// =======================================================================
// ARENA implementation of single source shortest path, which is essentially
// BFS-based.
//
// Author : Cheng Tan
//   Date : March 17, 2021

#include "../../lib/ARENA.h"
#include <iostream>
#include <fstream>
#include <string>

#define SIZE 16384
//#define SIZE 16

#define NODES 4

#define BFS_KERNEL 1

int local_rank;
int local_start;
int local_end;

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// ----------------------------------------------------------------------
int adj[SIZE/NODES][SIZE]; // 'adj[i][j] = SIZE' indicates i->j.
int path[SIZE/NODES]; // Results. Initially, path[i] = -1.

int adj_gold[SIZE][SIZE];
int path_gold[SIZE];

// ----------------------------------------------------------------------
// Initialize random local data value for the demo.
// ----------------------------------------------------------------------
void init_data(); // adjacency matrix can be read from an input file
bool verify(int);

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// ----------------------------------------------------------------------
void bfs_kernel(int start, int end, int param, bool require_data, int length) {
  int cur_node = start;
  int depth = param;

  // Ckeck connections from current node to all the neighbors.
  for(int neighbor=0; neighbor<SIZE; ++neighbor) {
    if(adj[cur_node][neighbor] > depth) {
      adj[cur_node][neighbor] = depth;

      // TODO: spawn task based on the neighbor's data range.
      // ARENA_spawn_task(BFS_KERNEL, /*neighbor_start*/, /*neighbor_end*/, /*next_depth*/);
      ARENA_spawn_task(BFS_KERNEL, neighbor, neighbor+1, depth+1);
    }
  }
  if(path[cur_node] == -1 or path[cur_node] > depth)
    path[cur_node] = depth;
}

// ----------------------------------------------------------------------
// Main function.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // initialize global data start and end
  local_rank = ARENA_init(NODES);

  // declare local data range
  local_start = local_rank * (SIZE / NODES);
  local_end   = (local_rank + 1) * (SIZE / NODES);
  ARENA_set_local(local_start, local_end);

  // register kernel
  bool isRoot    = true;
  int root_start = 1; //single source point
  int root_end   = 2;
  int root_param = 1;
  ARENA_register_task(BFS_KERNEL, &bfs_kernel, isRoot, root_start, root_end, root_param);
  // ARENA_register_task(ARENA_NORMAL_TASK0, &kernel0);
  // ARENA_register_task(ARENA_NORMAL_TASK1, &kernel1);
  // ARENA_register_task(ARENA_NORMAL_TASK2, &kernek2);

  // init local data
  init_data();

  // execute ARENA runtime 
  ARENA_run();

  // Verify
  if(verify(root_start)) {
    cout<<"[success] rank "<<local_rank<<endl;
  } else {
    cout<<"[fail] rank "<<local_rank<<endl;
  }

  return 0;
}


// ----------------------------------------------------------------------
// helper functions
// ----------------------------------------------------------------------
void init_data() {
  for(int i=0; i<SIZE/NODES; ++i) {
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

bool verify(int source) {
  for(int i=0; i<SIZE; ++i) {
    path_gold[i] = -1;
    for(int j=0; j<SIZE; ++j) {
      adj_gold[i][j] = 0;
    }
  }

  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      if(j == 2*i+1 or j == 2*i+2)
        adj_gold[i][j] = SIZE;
    }
  }

  chrono::system_clock::time_point start = chrono::system_clock::now();

  queue<int> bfs_q;
  bfs_q.push(source);
  int len = bfs_q.size();
  int level = 1;
  path_gold[source] = level;
  while(!bfs_q.empty()) {
    int node = bfs_q.front();
    level = path_gold[node] + 1;
    bfs_q.pop();
    for(int i=0; i<SIZE; ++i) {
      if(adj_gold[node][i] > level) {
        bfs_q.push(i);
        adj_gold[node][i] = level;
        path_gold[i] = level;
      }
    }
  }

  chrono::system_clock::time_point end = chrono::system_clock::now();
  chrono::duration<double> elapsed_seconds = end - start;
  if(local_rank == 0)
    cout<<"baseline execution time: "<<elapsed_seconds.count()<<"s"<<endl;

  for(int i=local_start; i<local_end; ++i) {
    if(path[i-local_start] != path_gold[i])
      return false;
  }

  return true;

}


