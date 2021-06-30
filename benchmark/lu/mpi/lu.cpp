// =======================================================================
// lu.cpp
// =======================================================================
// MPI implementation of LU 
//
// Author : Cheng Tan
//   Date : April 10, 2020

#include "mpi.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <string>

#define SIZE 1024
#define NODES 4

using namespace std;

long int total_data_in  = 0;
long int total_data_out = 0;

int flag_profile;
MPI_Request request_profile = MPI_REQUEST_NULL;
MPI_Status  status_profile;

int nodes;
int local_rank;
int local_start;
int local_end;
int local_bound;

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

void init_data() {
  int tmp = 0;
  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      A[i][j] = tmp;
      B[i][j] = 0;
      tmp++;
    }
    tmp = tmp - (SIZE - 1);
  }
}

void init_kernel() {
  int N = SIZE;
  int rank = local_rank;
  for(int i=0; i<N; ++i) {
    for(int j=0; j<i+1; ++j)
      A[i][j] = -(j % N) / (N*1.0) + 1.0;
    for(int j=i+1; j<N; ++j)
      A[i][j] = 0.0;
    A[i][i] = 1.0;
  }
  for(int t=0; t<N; ++t)
    for(int r=0; r<N; ++r)
      for(int s=0; s<N; ++s)
        B[r][s] += A[r][t] * A[s][t];
  for(int r=0; r<N; ++r)
    for(int s=0; s<N; ++s)
      A[r][s] = B[r][s];
  for(int i=local_start; i<local_end; ++i)
    for(int j=0; j<SIZE; ++j)
      local_A[i][j] = A[i][j];
//  cout<<"[init] rank "<<rank<<" local_A: "<<endl;
//  for(int i=0; i<SIZE; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      cout<<local_A[i][j]<<" ";
//    }
//    cout<<endl;
//  }
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// TODO: user specified.
// ----------------------------------------------------------------------
int lu_kernel() {
  int N = SIZE;
  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<i; ++j) {
      for(int k=0; k<j; ++k) {
        local_A[i][j] -= (local_A[i][k] * local_A[k][j]);
      }
      local_A[i][j] /= local_A[j][j];
    }
    for(int j=i; j<N; ++j) {
      for(int k=0; k<i; ++k) {
        local_A[i][j] -= (local_A[i][k] * local_A[k][j]);
      }
    }
  }
  MPI_Test(&request_profile, &flag_profile, &status_profile);
//  for(int x = 0; x < N * i; ++x) {
//    int j = x / i;
//    int k = x % i;
//    if(j < i) {
//      if(k < j) {
//        local_A[i][j] -= local_A[i][k] * local_A[k][j];
//      }
//      if(k == i - 1) {
//        local_A[i][j] /= local_A[j][j];
//      }
//    } else {
//      if(k < i) {
//        local_A[i][j] -= local_A[i][k] * local_A[k][j];
//      }
//    }
//  }

//  cout<<"[result] rank "<<local_rank<<" local_A: "<<endl;
//  for(int i=0; i<SIZE; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      cout<<local_A[i][j]<<" ";
//    }
//    cout<<endl;
//  }

  return 1;
}

// ----------------------------------------------------------------------
// Initialize task start point, data tag, and remote data requirement.
// TODO: user specified
// ----------------------------------------------------------------------
void init_task(int argc, char *argv[]) {
  // MPI initial
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

  init_data();

  local_bound = local_rank * (SIZE/NODES);
  local_start = local_rank * (SIZE/NODES);
  local_end   = local_rank * (SIZE/NODES) + SIZE/NODES;

  cout<<"rank "<<local_rank<<" nodes "<<nodes<<" local_start "<<local_start<<" local_end "<<local_end<<endl;
}

// ----------------------------------------------------------------------
// Main function. No need to change.
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
  if(local_rank > 0) {
    for(int i=0; i<local_rank*SIZE/NODES; ++i) {
      total_data_in += SIZE;
      MPI_Recv(local_A[i], SIZE, MPI_FLOAT, local_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  lu_kernel();
  if(local_rank < NODES-1) {
    for(int i=0; i<(local_rank+1)*SIZE/NODES; ++i) {
      total_data_out += SIZE;
      MPI_Send(local_A[i], SIZE, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
    }
  }

  chrono::system_clock::time_point end = chrono::system_clock::now();

  // Output
//  if(local_rank == NODES - 1) {
//    cout<<"[output] rank "<<local_rank<<endl;
//    for(int i=0; i<SIZE; ++i) {
//      for(int j=0; j<SIZE; ++j) {
//        cout<<local_A[i][j]<<" ";
//      }
//      cout<<endl;
//    }
//  }

  chrono::duration<double> elapsed_seconds = end-start;
  time_t end_time = chrono::system_clock::to_time_t(end);
  cout<<"[time] rank "<<local_rank<<" elapsed time: "
      <<elapsed_seconds.count()<<"s"<<endl;
       // " finished computation at " << ctime(&end_time)

  cout<<"[data movement] rank "<<local_rank<<" data in: "<<total_data_in<<" data out: "<<total_data_out<<" total: "<<total_data_in+total_data_out<<" size: "<<4*(total_data_in+total_data_out)<<endl;

  // MPI finalize
  MPI_Finalize();
  return 0;
}

