// =======================================================================
// spmv_mpi.cpp
// =======================================================================
// Conventional MPI implementation of SPMV.
//
// Author : Cheng Tan
//   Date : February 18, 2020

#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<iostream>

#define SIZE 4
#define BASELINE 1

using namespace std;

int spmv(int, int*, int*, int*, int);

int main(int argc, char *argv[])
{
  // Variables and Initializations
  int size = SIZE, rank;
  int A[SIZE][SIZE] = {{0,0,0,0}, {5,8,0,0}, {0,0,3,0}, {0,6,0,0}};
  int V[SIZE] = {5,8,3,6};
  int COL_INDEX[SIZE] = {0,1,2,1};
  int ROW_INDEX[SIZE+1] = {0,0,2,3,4};
  int* x = new int[SIZE];
  int* y;

  // MPI Code
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request request = MPI_REQUEST_NULL;

  // Local data
  for(int i=0; i<SIZE; ++i)
    x[i] = i+1;
  int local_row_start = ROW_INDEX[rank];
  int local_row_end = ROW_INDEX[rank+1];
  int local_row_range = local_row_end - local_row_start;
  int* local_V = new int[local_row_range];
  for(int i=local_row_start; i<local_row_end; ++i)
    local_V[i-local_row_start] = V[i];
  int* local_COL = new int[local_row_range];
  for(int i=local_row_start; i<local_row_end; ++i)
    local_COL[i-local_row_start] = COL_INDEX[i];
  int* local_ROW = new int[2];
  local_ROW[0] = local_row_start;
  local_ROW[1] = local_row_end;
  // TODO: assume size==SIZE for now
  int local_x_range = size/SIZE;
  int* local_x = new int[local_x_range];
  for(int i=0; i<local_x_range; ++i) {
    local_x[i] = x[i+local_x_range*rank];
  }
  int* remote_send_count = new int[size];
  int* remote_recv_count = new int[size];
  int** remote_COL = new int*[size];
  int** remote_V = new int*[size];
  int* remote_row_range = new int[size];
  for(int i=0; i<size; ++i) {
    remote_send_count[i] = 0;
    remote_recv_count[i] = 0;
    remote_COL[i] = NULL;
    remote_V[i] = NULL;
    remote_row_range[i] = 0;
  }

  // Local computation on local data
  int local_output = spmv(local_row_range, local_V, local_COL, local_x, rank);
  cout<<"rank "<<rank<<" local output: "<<local_output<<endl;
  cout<<"-------------------"<<endl;

  // Transfer data
  for(int i=0; i<local_row_range; ++i) {
    // TODO: this judgement should be replaced by a function,
    remote_send_count[local_COL[i]]++;
  }
  for(int i=0; i<size; ++i) {
    if(i!=rank) {
      MPI_Isend(&remote_send_count[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
      MPI_Irecv(&remote_recv_count[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
    }
  }
  MPI_Status status;
  MPI_Wait(&request, &status);
  for(int i=0; i<size; ++i) {
    if(i!=rank) {
      cout<<"rank ["<<rank<<"] recv from core ["<<i<<"] of value "<<remote_recv_count[i]<<endl;
    }
  }
  cout<<"-------------------"<<endl;
  int current_sent = 0;
  for(int i=0; i<size; ++i) {
//    while(remote_send_count[i]>0) {
    for(int j=0; j<local_row_range; ++j) {
      // TODO: this should be replaced by a function to match the dest,
      //       e.g., match(i, local_COL[i])
      if(i==local_COL[j] and i!=rank) {
        MPI_Isend(&local_COL[j], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
        MPI_Isend(&local_V[j], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
        cout<<"[cheng] rank "<<rank<<" dest "<<i<<" local_COL: "<<local_COL[j]<<" local_V: "<<local_V[j]<<endl;
//        remote_send_count[i]--;
      }
    }
    if(remote_recv_count[i] != 0) {
      remote_COL[i] = new int[remote_recv_count[i]];
      remote_V[i] = new int[remote_recv_count[i]];
      for(int j=0; j<remote_recv_count[i]; ++j) {
        cout<<"rank: "<<rank<<" recv from "<<i<<" remote_recv_count[i]: "<<remote_recv_count[i]<<endl;
        MPI_Irecv(&remote_COL[i][0], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
        MPI_Irecv(&remote_V[i][0], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
        remote_row_range[i]++;
      }
    }
  }
  MPI_Wait(&request, &status);

  for(int i=0; i<size; ++i) {
    if(remote_recv_count[i]!=0) {
      cout<<"rank ["<<rank<<"] recv from core ["<<i<<"] of col "<<remote_COL[i][0]<<"; value "<<remote_V[i][0]<<endl;
    }
  }
  cout<<"-------------------"<<endl;

  // Local computation on remote data
  int remote_output = 0;
  for(int i=0; i<size; ++i) {
    if(remote_row_range[i] != 0) {
      remote_output += spmv(remote_row_range[i], remote_V[i], remote_COL[i], local_x, rank);
      cout<<"********** rank "<<rank<<" accumulated remote output: "<<remote_output<<" from proc "<<i<<endl;
    }
  }

  //End of BFS code
  MPI_Finalize();
  return 0;
}

int spmv(int local_row_range, int* local_V, int* local_COL, int* local_x, int local_x_index) {
  int output = 0;
  for(int i=0; i<local_row_range; ++i) {
    // TODO: need a justification function here
    if(local_COL[i]==local_x_index) {
      // TODO: need provide accurate index
      output += local_V[i] * local_x[0];
    }
  }
  return output;
}
