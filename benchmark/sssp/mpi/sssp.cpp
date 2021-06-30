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
#include <fstream>
#include <chrono>
#include <ctime> 
#include <string>

//#define DEBUG

//#define SIZE 65536
//#define SIZE 4096
#define SIZE 16384
//#define NODES 2
#define NODES 4
//#define NODES 8
//#define NODES 16

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

long int total_data_in  = 0;
long int total_data_out = 0;

int flag_profile;
MPI_Request request_profile = MPI_REQUEST_NULL;
MPI_Status  status_profile;
// ----------------------------------------------------------------------
// Total data allocated onto nodes.
// TODO: user specified.
// ----------------------------------------------------------------------
//int GRAPH[SIZE][SIZE];

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
int** M;
//int M[SIZE/NODES][SIZE+1];
int num_vertice;
int num_edge;
int* vertices;
int* edges;
int* visited;

void init_data() {

  M = new int*[SIZE/NODES];
  for(int i=0; i<SIZE/NODES; ++i) {
    M[i] = new int[SIZE+1];
    for(int j=0; j<SIZE+1; ++j) {
      M[i][j] = 0;
    }
  }
//  int count;
//  ifstream File;
//  File.open("../data/graph4096.txt");
////  File.open("../data/graph65536.txt");
//  File >> num_vertice;
//  vertices = new int[num_vertice + 1];
//  visited = new int[num_vertice];
//  string line;
//  std::string::size_type sz;   // alias of size_t
//  for(int i=0; i<num_vertice; ++i) {
//    File >> line;
//    vertices[i] = stoi(line, &sz);
//    visited[i] = num_vertice+1;
//    File >> line;
//  }
//
//  File >> line;
//  global_start = std::stoi(line, &sz);
//
//  File >> num_edge;
//  vertices[num_vertice] = num_edge;
//  edges = new int[num_edge];
//  for(int i=0; i<num_edge; ++i) {
//    File >> line;
//    edges[i] = stoi(line, &sz);
//    File >> line;
//  }
//
//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=vertices[i]; j<vertices[i+1]; ++j) {
//      M[i-local_bound][edges[j]] = SIZE;
////      GRAPH[i][edges[j]] = SIZE;
//    }
//    M[i-local_bound][SIZE] = SIZE;
//  }

  global_start = 1;

//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      if(j == 4*i+1 or j == 4*i+2 or j == 4*i+3 or j == 4*i+4)
//        M[i-local_bound][j] = SIZE;
//    }
//    M[i-local_bound][SIZE] = SIZE;
//  }

//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      if(j == 3*i+1 or j == 3*i+2 or j == 3*i+3)
//        M[i-local_bound][j] = SIZE;
//    }
//    M[i-local_bound][SIZE] = SIZE;
//  }

  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      if(j == 2*i+1 or j == 2*i+2)
        M[i-local_bound][j] = SIZE;
    }
    M[i-local_bound][SIZE] = SIZE;
  }

//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      if(j == i+1)
//        M[i-local_bound][j] = SIZE;
//    }
//    M[i-local_bound][SIZE] = SIZE;
//  }

}

void init_kernel() {

//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      M[i-local_bound][j] = GRAPH[i][j];
//    }
//    M[i-local_bound][SIZE] = SIZE;
//  }
  cout<<"[init] rank "<<local_rank<<" : ";
//  for(int i=0; i<SIZE; ++i) {
//    M[0][i] = GRAPH[local_rank][i];
//    cout<<M[0][i]<<" ";
//  }
//  cout<<endl;
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
//  MPI_Wait(&request_profile, &status_profile);
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
  MPI_Test(&request_profile, &flag_profile, &status_profile);
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
  local_bound = local_rank * (SIZE/NODES);
  local_start = local_rank * (SIZE/NODES);
  local_end   = local_rank * (SIZE/NODES) + SIZE/NODES;

  init_data();

  cout<<"rank "<<local_rank<<" nodes "<<nodes<<" local_start "<<local_start<<" local_end "<<local_end<<endl;
  if(local_start <= global_start and global_start < local_end) {
    num_ready_points = 1;
    ready_points[local_rank][0] = global_start;
  }
}

void synch() {
  total_data_in  += NODES;
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
#ifdef DEBUG
      for(int x=0; x<num_points_all[i]; ++x)
        cout<<"[bcast] rank "<<local_rank<<" sending point "<<ready_points[local_rank][x]<<" root "<<i<<endl;
#endif
      MPI_Bcast(ready_points[i], num_points_all[i], MPI_INT, i, MPI_COMM_WORLD);
      if(i == local_rank) {
        total_data_out += num_points_all[i]*NODES;
      } else {
        total_data_in += num_points_all[i];
      }
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

  chrono::system_clock::time_point start = chrono::system_clock::now();

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

//  MPI_Barrier(MPI_COMM_WORLD);

  chrono::system_clock::time_point end = chrono::system_clock::now();

  // Output
//  if(local_rank == NODES-1) {
//  cout<<"[final] rank "<<local_rank<<endl;
//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=SIZE; j<SIZE+1; ++j) {
//      cout<<M[i-local_bound][j]<<" ";
//    }
//    cout<<endl;
//  }}

  chrono::duration<double> elapsed_seconds = end-start;
  time_t end_time = chrono::system_clock::to_time_t(end);
  cout<<"[time] rank "<<local_rank<<" elapsed time: "
      <<elapsed_seconds.count()<<"s"<<endl;
       // " finished computation at " << ctime(&end_time)
  cout<<"[data movement] rank "<<local_rank<<" data in: "<<total_data_in<<" data out: "<<total_data_out<<" total: "<<total_data_in+total_data_out<<" size: "<<4*(total_data_in+total_data_out)<<endl;

  MPI_Finalize();
  return 0;
}

