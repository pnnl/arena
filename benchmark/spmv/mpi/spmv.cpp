// =======================================================================
// spmv.h
// =======================================================================
// MPI implementation of SPMV.
//
// Author : Cheng Tan
//   Date : April 10, 2020

#include "mpi.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <string>

#define TOTAL 4000
#define SIZE TOTAL*TOTAL/4 // Assume 25% sparsity.
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
int global_A[TOTAL][TOTAL];// = {{0,0,0,0}, {5,8,0,0}, {0,0,3,0}, {0,6,0,0}};
int global_V[SIZE]; // = {5,8,3,6};
int global_COL[SIZE]; // = {0,1,2,1};
int global_ROW[TOTAL+1]; // = {0,0,2,3,4};
int global_result[TOTAL];

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
int local_V[SIZE/NODES]; // = {0};
int local_COL[SIZE/NODES]; // = {0};
int local_X[TOTAL]; // = {1,2,3,4};
int local_OUT[TOTAL];
int local_RESULT[TOTAL];

void init_data() {
  for(int i=0; i<TOTAL; ++i) {
    for(int j=0; j<TOTAL; ++j) {
      if((i+j)%4 == 0)
        global_A[i][j] = i+j+1;
      else
        global_A[i][j] = 0;
    }
  }
  int k=0;
  global_ROW[0] = 0;
  for(int i=0; i<TOTAL; ++i) {
    for(int j=0; j<TOTAL; ++j) {
      if(global_A[i][j] != 0) {
        global_V[k] = global_A[i][j];
        global_COL[k] = j;
        k++;
      }
    }
    global_ROW[i+1] = k;
  }
  for(int i=0; i<TOTAL; ++i) {
    local_X[i] = i + 1;
    global_result[i] = 0;
  }
//  cout<<"global_ROW: "<<endl;
//  for(int i=0; i<TOTAL; ++i) {
//    cout<<" "<<global_ROW[i];
//  }
//  cout<<endl;
}

void init_kernel() {
  for(int i=0; i<SIZE/NODES; ++i) {
    local_V[i] = global_V[local_rank*SIZE/NODES+i];
    local_COL[i] = global_COL[local_rank*SIZE/NODES+i];
  }
  for(int i=0; i<TOTAL; ++i) {
    local_OUT[i] = 0;
  }
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// TODO: user specified.
// ----------------------------------------------------------------------
int spmv_kernel(int start, int end) {
  for(int i=0; i<end-start; ++i) {
    int result_index = 0;
    for(int j=1; j<TOTAL+1; ++j) {
      if(i+start < global_ROW[j]) {
        result_index = j-1;
        break;
      }
    }
    local_OUT[result_index] += local_V[i] * local_X[local_COL[i]];
  }
  MPI_Test(&request_profile, &flag_profile, &status_profile);
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

int main(int argc, char *argv[]) {

  // Initialize global data start and end
  init_task(argc, argv);

  // Initialize local allocated data
  init_kernel();

  if(nodes != NODES) return -1;

//  MPI_Barrier(MPI_COMM_WORLD);
  chrono::system_clock::time_point start = chrono::system_clock::now();

  // Execution
  spmv_kernel(local_start, local_end);
//  for(int i=0; i<NODES; ++i) {
//    MPI_Bcast(local_OUT, TOTAL, MPI_INT, i, MPI_COMM_WORLD);
//  }
  MPI_Reduce(local_OUT, local_RESULT, TOTAL, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(local_rank == 0) {
    total_data_in += TOTAL * NODES;
  } else {
    total_data_out += TOTAL;
  }

  chrono::system_clock::time_point end = chrono::system_clock::now();

// Output
//  if(local_rank == 0) {
//    cout<<"[final] rank "<<local_rank<<endl;
//    for(int i=0; i<TOTAL; ++i) {
//      cout<<local_RESULT[i]<<" ";
//    }
//    cout<<endl;
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

