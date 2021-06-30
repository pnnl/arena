// =======================================================================
// spmv.h
// =======================================================================
// ARENA implementation of SPMV.
//
// Author : Cheng Tan
//   Date : March 18, 2020

#include "mpi.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <string>

#define SIZE 1024
#define NODES 4
//#define NODES 16

using namespace std;

int flag_profile;
MPI_Request request_profile = MPI_REQUEST_NULL;
MPI_Status  status_profile;

int nodes;
int local_rank;
int local_start;
int local_end;
int local_bound;
int global_start;
int global_level;

long int total_data_in  = 0;
long int total_data_out = 0;

// ----------------------------------------------------------------------
// Total data allocated onto nodes TODO: user specified 
// ----------------------------------------------------------------------
int A[SIZE][SIZE];//   = {{1,2,3,4}, {2,3,4,5}, {3,4,5,6}, {4,5,6,7}};
int B[SIZE][SIZE];//   = {{1,2,1,2}, {1,2,1,2}, {1,2,1,2}, {1,2,1,2}};
int C[SIZE][SIZE];//   = {{1,2,1,2}, {1,2,1,2}, {1,2,1,2}, {1,2,1,2}};

// ----------------------------------------------------------------------
// Local data allocated onto a node TODO: user specified
// ----------------------------------------------------------------------
int local_A[SIZE/NODES][SIZE];//   = {{0,0,0,0}};
int local_B[SIZE][SIZE];// = {{0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}};
int local_C[SIZE/NODES][SIZE];//   = {{0,0,0,0}};
int local_OUT[SIZE/NODES][SIZE];// = {{0,0,0,0}};
int alpha = 2;
int beta  = 1;
int I = SIZE;
int J = SIZE;
int K = SIZE;

void init_data() {
  int tmp = 0;
  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      A[i][j] = tmp;
      B[i][j] = j;
      C[i][j] = i;
      tmp++;
    }
    tmp = tmp - (SIZE - 1);
  }
}

void init_kernel() {
  for(int i=local_start; i<local_end; ++i) {
    for(int j=0; j<SIZE; ++j) {
      local_A[i-local_bound][j] = A[i][j];
      if(local_rank == i/(SIZE/NODES))
        local_B[i][j] = B[j][i];
      local_C[i-local_bound][j] = C[i][j];
      local_OUT[i-local_bound][j] = 0;
    }
  }
//  cout<<"[init] rank "<<local_rank<<endl;
//  for(int i=local_start; i<local_end; ++i) {
//    for(int j=0; j<SIZE; ++j) {
//      cout<<local_B[i][j]<<" ";
//    }
//    cout<<endl;
//  }
}

// ----------------------------------------------------------------------
// Total data allocated onto nodes TODO: user specified
// ----------------------------------------------------------------------
void init_task(int argc, char *argv[]) {
  // MPI initial
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

  // TODO: Data tag.
  init_data();

  local_bound = local_rank * (SIZE/NODES);
  local_start = local_rank * (SIZE/NODES);
  local_end   = local_rank * (SIZE/NODES) + SIZE/NODES;

  cout<<"rank "<<local_rank<<" nodes "<<nodes<<" local_start "<<local_start<<" local_end "<<local_end<<endl;
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc) TODO: user specified
// ----------------------------------------------------------------------
int gemm_kernel(int col) {
//  row -= local_bound;
  for(int row=0; row<SIZE/NODES; ++row) {
    for(int i=0; i<SIZE; ++i) {
      local_OUT[row][col] += alpha * local_A[row][i] * local_B[col][i];
    }
    local_OUT[row][col] += local_C[row][col]*beta;
  }
  MPI_Test(&request_profile, &flag_profile, &status_profile);
//  cout<<"[OUT]";
//  for(int x=0; x<SIZE; ++x) {
//    cout<<" "<<OUT[0][x];
//  }
//  cout<<endl;
  return -1;
}

int main(int argc, char *argv[]) {

  // Initialize global data start and end
  init_task(argc, argv);

  // Initialize local allocated data
  init_kernel();
  int level = 1;
  if(nodes != NODES) return -1;

  chrono::system_clock::time_point start = chrono::system_clock::now();

  // Execution
  for(int i=0; i<SIZE; ++i) {
    MPI_Bcast(local_B[i], SIZE, MPI_INT, i/(SIZE/NODES), MPI_COMM_WORLD);
    if(i/(SIZE/NODES) == local_rank) {
      total_data_out += SIZE*NODES;
    } else {
      total_data_in += SIZE;
    }
    gemm_kernel(i);
  }

  chrono::system_clock::time_point end = chrono::system_clock::now();

  // Output
//  if(local_rank ==0) {
//    cout<<"[output] rank "<<local_rank<<endl;
//    for(int i=local_start; i<local_end; ++i) {
//      for(int j=0; j<SIZE; ++j) {
//        cout<<local_OUT[i-local_bound][j]<<" ";
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

