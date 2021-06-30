// =======================================================================
// bfs.cpp
// =======================================================================
// Conventional BFS implementation.
//
// Author : Cheng Tan
//   Date : March 29, 2020

#include "mpi.h"
#include <queue>
#include <iostream>
#include <chrono>
#include <ctime> 

//#define DEBUG

//#define SIZE 8000
#define SIZE 8
#define NODES 4

using namespace std;

queue<int> send_queue;
queue<int> recv_queue;

int local_start;
int local_end;
int local_bound;
int global_start;
int global_level;

int nodes;
int local_rank;

int num_points_all[NODES];
int ready_points[NODES][SIZE];
int num_ready_points = 0;

// ----------------------------------------------------------------------
// Total data allocated onto nodes.
// TODO: user specified.
// ----------------------------------------------------------------------
int GRAPH[SIZE][SIZE];// = {{0,SIZE,0,0}, {0,0,SIZE,0}, {0,0,0,SIZE}, {0,0,0,0}};
//int GRAPH[SIZE][SIZE] = {{0,0,0,SIZE}, {SIZE,0,SIZE,0}, {0,0,0,0}, {0,SIZE,0,0}};
//int GRAPH[SIZE][SIZE] = {{0,0,0,SIZE,0,0,0,0},    {SIZE,0,SIZE,0,SIZE,SIZE,0,0},
//                         {0,0,0,SIZE,0,0,SIZE,0}, {0,SIZE,0,0,0,0,0,0},
//                         {0,0,0,0,0,0,0,0},       {0,0,0,0,0,0,0,SIZE},
//                         {0,0,0,0,0,0,0,SIZE},    {0,0,0,0,0,0,0,0}             };

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
int M[SIZE/NODES][SIZE+1];
//int M[SIZE/NODES][SIZE+1] = {{0,0,0,0,SIZE}};
//int M[SIZE/NODES][SIZE+1] = {{0,0,0,0,0,0,0,0,SIZE}, {0,0,0,0,0,0,0,0,SIZE}};
void init_kernel() {
  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      if(j == i + 1)
        GRAPH[i][j] = SIZE;
    }
  }
  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      M[i-local_bound][j] = GRAPH[i][j];
    }
    M[i-local_bound][SIZE] = SIZE;
  }
  cout<<"[init] rank "<<local_rank<<" : ";
  for(int i=0; i<SIZE; ++i) {
//    M[0][i] = GRAPH[local_rank][i];
    cout<<M[0][i]<<" ";
  }
  cout<<endl;
//  for(int i=0; i<SIZE; ++i) {
////    M[0][i] = GRAPH[local_rank][i];
//    cout<<M[1][i]<<" ";
//  }
//  cout<<endl;
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// TODO: user specified.
// ----------------------------------------------------------------------
void BFS_kernel(int point, int level) {
  for(int j=0; j<SIZE; ++j) {
    if(M[point][j] > level) {
      M[point][j] = level;
      ready_points[local_rank][num_ready_points++] = j;
#ifdef DEBUG
      cout<<"[result] rank "<<local_rank<<" point "<<point+local_bound<<" level "<<level<<" detected "<<j<<endl;
#endif
    }
  }
  if(M[point][SIZE] > level)
    M[point][SIZE] = level;

}

void init_task(int argc, char *argv[]) {
  // MPI initial
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

  // TODO: Task start point.
  global_start = 1;
  global_level = 1;

  // TODO: Data tag.
//  local_bound = local_rank * 2;
//  local_start = local_rank * 2;
//  local_end   = local_rank * 2 + 2;

  local_bound = local_rank * (SIZE/NODES);
  local_start = local_rank * (SIZE/NODES);
  local_end   = local_rank * (SIZE/NODES) + SIZE/NODES;

  cout<<"rank "<<local_rank<<" nodes "<<nodes<<" local_start "<<local_start<<" local_end "<<local_end<<endl;
  if(local_start <= global_start and global_start < local_end) {
    num_ready_points = 1;
    ready_points[local_rank][0] = global_start;
  }
}

void synch() {
  MPI_Allgather(&num_ready_points, 1, MPI_INT, num_points_all, 1, MPI_INT,
                MPI_COMM_WORLD);
#ifdef DEBUG
  cout<<"rank "<<local_rank<<" num_points_all "<<endl;
  for(int i=0; i<NODES; ++i) {
    cout<<num_points_all[i]<<" ";
  }
  cout<<endl;
  cout<<"[ready] rank "<<local_rank<<" ready_points "<<endl;
  for(int i=0; i<num_points_all[local_rank]; ++i) {
    cout<<ready_points[local_rank][i]<<" ";
  }
  cout<<endl;
#endif
  for(int i=0; i<NODES; ++i) {
    if(num_points_all[i] > 0) {
      for(int x=0; x<num_points_all[i]; ++x)
#ifdef DEBUG
        cout<<"[bcast] rank "<<local_rank<<" sending point "<<ready_points[local_rank][x]<<" root "<<i<<endl;
#endif
      MPI_Bcast(ready_points[i], num_points_all[i], MPI_INT, i, MPI_COMM_WORLD);
      for(int j=0; j<num_points_all[i]; ++j) {
        recv_queue.push(ready_points[i][j]);
      }
#ifdef DEBUG
      cout<<"rank "<<local_rank<<" received #points: "<<num_points_all[i]<<" from "<<i<<endl;
#endif
    }
  }
  num_ready_points = 0;
}

// ----------------------------------------------------------------------
// Main function.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // Initialize global data start and end
  init_task(argc, argv);

  // Initialize local allocated data
  init_kernel();
  int level = 1;
  if(nodes != NODES) return -1;

  auto start = chrono::system_clock::now();

  // Execution
  while(1) {

    synch();
    if(recv_queue.empty())
      break;
    while(!recv_queue.empty()) {
      int point = recv_queue.front();
#ifdef DEBUG
      cout<<"[checking] rank "<<local_rank<<" is checking point "<<point<<" size "<<recv_queue.size()<<endl;
#endif
      recv_queue.pop();
      // Filter out the points on this node.
      if(local_start <= point and point < local_end)
        BFS_kernel(point-local_bound, level);
    }
    ++level;
  }

  auto end = chrono::system_clock::now();

  // Output
  cout<<"[final] rank "<<local_rank<<endl;
  for(int i=local_start; i<local_end; ++i) {
    for(int j=SIZE; j<SIZE+1; ++j) {
      cout<<M[i-local_bound][j]<<" ";
    }
    cout<<endl;
  }

  chrono::duration<double> elapsed_seconds = end-start;
  time_t end_time = chrono::system_clock::to_time_t(end);
  cout<<"[time] rank "<<local_rank<<" elapsed time: "
      <<elapsed_seconds.count()<<"s"<<endl;
       // " finished computation at " << ctime(&end_time)

  MPI_Finalize();
  return 0;
}

