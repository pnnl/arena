// =======================================================================
// bfs.cpp
// =======================================================================
// Conventional BFS implementation.
//
// Author : Cheng Tan
//   Date : March 29, 2020

#include <queue>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime> 
#include <string>

//#define DEBUG

//#define SIZE 8000
//#define SIZE 65536
//#define SIZE 37768
//#define SIZE 18884
#define SIZE 16384
//#define SIZE 40
//#define NODES 16
#define NODES 1

using namespace std;

queue<int> send_queue;
queue<int> recv_queue;
queue<int> ready_points;

int local_start;
int local_end;
int local_bound;
int global_start;
int global_level;

int nodes;
int local_rank;

//int num_points_all[NODES];
//int num_ready_points = 0;

// ----------------------------------------------------------------------
// Total data allocated onto nodes.
// TODO: user specified.
// ----------------------------------------------------------------------
//int GRAPH[SIZE][SIZE];

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
int** M;//[SIZE/NODES][SIZE+1];
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
//  for(int i=0; i<SIZE; ++i) {
//    for(int j=vertices[i]; j<vertices[i+1]; ++j) {
//      GRAPH[i][edges[j]] = SIZE;
//    }
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

//  for(int i=0; i<SIZE; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      if(j == i+1)
//        GRAPH[i][j] = SIZE;
//    }
//  }

}

void init_kernel() {

//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      M[i-local_bound][j] = GRAPH[i][j];
//    }
//    M[i-local_bound][SIZE] = SIZE;
//  }

//  cout<<"[init] rank "<<local_rank<<" : ";

//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=0; j<SIZE; ++j) {
////    M[0][i] = GRAPH[local_rank][i];
//      cout<<M[i][j]<<" ";
//    }
//    cout<<endl;
//  }
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// TODO: user specified.
// ----------------------------------------------------------------------
void BFS_kernel(int point, int level) {
  for(int j=0; j<SIZE; ++j) {
//    cout<<"step 5.1.1... point: "<<point<<" j: "<<j<<endl;
    if(M[point][j] > level) {
//      cout<<"step 5.1.2... point: "<<point<<" j: "<<j<<" num_ready_points: "<<ready_points.size()<<endl;
      M[point][j] = level;
      ready_points.push(j);
#ifdef DEBUG
      cout<<"[result] rank "<<local_rank<<" point "<<point+local_bound<<" level "<<level<<" detected "<<j<<endl;
#endif
    }
  }
  if(M[point][SIZE] > level)
    M[point][SIZE] = level;

}

void init_task(int argc, char *argv[]) {

  nodes = NODES;
  local_rank = 0;

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
    ready_points.push(global_start);
  }
}

void synch() {
  while(ready_points.size()>0) {
    recv_queue.push(ready_points.front());
    ready_points.pop();
  }
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

  int sst_me = 0;
chrono::system_clock::time_point sst_start;
chrono::system_clock::time_point sst_end;
  chrono::system_clock::time_point start = chrono::system_clock::now();

  // Execution
  while(1) {

//    cout<<"step 2..."<<endl;
    synch();
//    cout<<"step 3..."<<endl;
    if(recv_queue.empty())
      break;
//    cout<<"step 4..."<<endl;
    while(!recv_queue.empty()) {
//      cout<<"step 5..."<<endl;
      int point = recv_queue.front();
#ifdef DEBUG
      cout<<"[checking] rank "<<local_rank<<" is checking point "<<point<<" size "<<recv_queue.size()<<endl;
#endif
      recv_queue.pop();
  if(sst_me == 0){
    sst_start = chrono::system_clock::now();
  }
      // Filter out the points on this node.
      BFS_kernel(point, level);
//      cout<<"step 6..."<<endl;
//
  if(sst_me == 0){
    sst_end = chrono::system_clock::now();
    sst_me = 1;
  }
    }
    ++level;
  }

  chrono::system_clock::time_point end = chrono::system_clock::now();

  // Output
//  if(local_rank ==0) {
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
       //
  chrono::duration<double> sst_elapsed_seconds = sst_end-sst_start;
  cout<<"[time] rank "<<local_rank<<" elapsed time: "
      <<sst_elapsed_seconds.count()<<"s"<<endl;


  return 0;
}

