// =======================================================================
// spmv.cpp
// =======================================================================
// Serial implementation of SPMV.
//
// Author : Cheng Tan
//   Date : February 18, 2020

#include<stdio.h>
#include<stdlib.h>
#include<iostream>

#define SIZE 4
#define BASELINE 1

using namespace std;

int* spmv_simple(int t_A[][SIZE], int t_x[SIZE]) {
  int* y = new int[SIZE];
  int temp_out = 0;
  for(int i=0; i<SIZE; ++i) {
    temp_out = 0;
    for(int j=0; j<SIZE; ++j) {
      temp_out += t_A[i][j] * t_x[j];
    }
    y[i] = temp_out;
  }
  return y;
}

int* spmv_csr(int t_V[SIZE], int t_COL[SIZE], int t_ROW[SIZE], int t_x[SIZE]) {
  int* y = new int[SIZE];
  int temp_out = 0;
  int row_start = 0;
  int row_end = 0;
  int out = 0;
  for(int r_i=0; r_i<SIZE; ++r_i) {
    row_start = t_ROW[r_i];
    row_end = t_ROW[r_i+1];
    out = 0;
    for(int v_i=0; v_i<row_end-row_start; ++v_i) {
      out += t_V[v_i+row_start] * t_x[t_COL[v_i+row_start]];
    }
    y[r_i] = out;
  }
  return y;
}

void print(int* t_y) {
  for(int i=0; i<SIZE; ++i) {
    cout<<t_y[i]<<" ";
  }
  cout<<endl;
}

int main(int argc, char *argv[]) {
  int A[SIZE][SIZE] = {{0,0,0,0}, {5,8,0,0}, {0,0,3,0}, {0,6,0,0}};
  int V[SIZE] = {5,8,3,6};
  int COL_INDEX[SIZE] = {0,1,2,1};
  int ROW_INDEX[SIZE+1] = {0,0,2,3,4};
  int x[SIZE] = {1,2,3,4};
  int* y;

  y = spmv_simple(A, x); 
  print(y);
  y = spmv_csr(V, COL_INDEX, ROW_INDEX, x);
  print(y);
  return 0;
}
