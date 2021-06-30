// ======================================================================
// gcn_2_layer_csr.cpp
// ======================================================================
// Single layer GCN's Ax implementation, which can be viewed as GEMV.
// Note that the entire implementation of single layer GCN is FC(Ax, W).
//
// Author : Cheng Tan
//   Date : Oct 10, 2020

#include "mpi.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime> 
#include <string>

//#define DEBUG
// Note that the vectorization version works for single core sequential
// implementation, instead of multiple-node MPI version, which is due
// to the number of nodes in the graph is not dividable by 8.
//#define VECTORIZATION
//#define DUMMY_DATA


//#define num_vertice 2708
//#define num_feature 1433
//#define num_w0_out 16
//#define num_w1_out 7

//#define num_vertice 2708
//#define num_feature 200
//#define num_w0_out 16
//#define num_w1_out 7

//#define num_vertice 8000
//#define NODES 1
#define NODES 4
//#define NODES 8
//#define NODES 16

#define BLK_SIZE 20

using namespace std;

int local_start;
int local_end;
int local_bound;
int global_start;

int nodes;
int local_rank;

long int total_data_in  = 0;
long int total_data_out = 0;

int flag_profile;
MPI_Request request_profile = MPI_REQUEST_NULL;
MPI_Status  status_profile;

int num_vertice;
int num_nonzero_A = 0;
int num_feature;
int num_w0_out;
int num_w1_out;
int** global_A;
float** local_A;
float** global_X;
float** global_weight0;
float* global_bias0;
float** global_weight1;
float* global_bias1;
float** output_gold;

int local_nnz;
int local_nnb = 1;
int blk_size;
float* local_V_COO;
int* local_COL_COO;
int* local_ROW_COO;
float* local_V_HiCOO;
//int* temp_local_COL_HiCOO;
//int* temp_local_ROW_HiCOO;
int* temp_local_BLK_COL_HiCOO;
int* temp_local_BLK_ROW_HiCOO;

char* local_COL_HiCOO;
char* local_ROW_HiCOO;
int* local_BLK_COL_HiCOO;
int* local_BLK_ROW_HiCOO;
int* local_BLK_SIZE_HiCOO;

float** local_X;
float* communicate_buffer;
float* temp_communicate_buffer;
float* current_feature;
float** out0;
float** out1;
float** out2;
float** out3;

void init_data() {

#ifdef DUMMY_DATA
  num_vertice = 8;
  global_A = new int*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_A[i] = new int[num_vertice];
    for(int j=0; j<num_vertice; ++j) {
      global_A[i][j] = 0;
    }
  }
  for(int i=0; i<num_vertice; ++i) {
    for(int j=i; j<num_vertice; ++j) {
      global_A[i][j] = j%2;
      global_A[j][i] = j%2;
    }
  }

  num_feature = 4;

  global_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      global_X[i][j] = i*num_feature + j;
    }
  }

  num_w0_out = 2;

  global_weight0 = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    global_weight0[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      global_weight0[i][j] = i;
    }
  }

  global_bias0 = new float[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    global_bias0[i] = 0;
  }

  num_w1_out = 1;

  global_weight1 = new float*[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    global_weight1[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      global_weight1[i][j] = i*num_w1_out + j;
    }
  }

  global_bias1 = new float[num_w1_out];
  for(int i=0; i<num_w1_out; ++i) {
    global_bias1[i] = 0;
  }

  output_gold = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    output_gold[i] = new float[num_w1_out];
  }
//  output_gold[0][0] = 338;
//  output_gold[1][0] = 462;
//  output_gold[2][0] = 200;
//  output_gold[3][0] = 548;
  output_gold[0][0] = 2380;
  output_gold[1][0] = 2820;
  output_gold[2][0] = 1926;
  output_gold[3][0] = 3222;
  output_gold[4][0] = 1410;
  output_gold[5][0] = 3538;
  output_gold[6][0] = 784;
  output_gold[7][0] = 3720;

#endif
#ifndef DUMMY_DATA
  // read data from files and initialize input data arrays
  ifstream File;
  File.open("../../data/cora_2layer/input_a");
  File >> num_vertice;

  global_A = new int*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_A[i] = new int[num_vertice];
    for(int j=0; j<num_vertice; ++j) {
      File >> global_A[i][j];
      if(global_A[i][j] != 0) {
        num_nonzero_A += 1;
      }
    }
  }
//  cout<<"see "<<num_nonzero_A<<endl;
  File.close();

  File.open("../../data/cora_2layer/input_x");
  File >> num_feature;

  global_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      File >> global_X[i][j];
    }
  }
  File.close();

  File.open("../../data/cora_2layer/input_w0");
  File >> num_w0_out;

  global_weight0 = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    global_weight0[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      File >> global_weight0[i][j];
    }
  }

  global_bias0 = new float[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    File >> global_bias0[i];
  }
  File.close();

  File.open("../../data/cora_2layer/input_w1");
  File >> num_w1_out;

  global_weight1 = new float*[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    global_weight1[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      File >> global_weight1[i][j];
    }
  }

  global_bias1 = new float[num_w1_out];
  for(int i=0; i<num_w1_out; ++i) {
    File >> global_bias1[i];
  }
  File.close();

  File.open("../../data/cora_2layer/output_gold");

  output_gold = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    output_gold[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      File >> output_gold[i][j];
    }
  }
  File.close();

#endif

  local_bound = num_vertice/NODES;

//  global_A[0][4] = 1;
  local_A = new float*[num_vertice/NODES];
  local_X = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    local_A[i] = new float[num_vertice];
    for(int j=0; j<num_vertice; ++j) {
      local_A[i][j] = global_A[local_rank*local_bound+i][j];
    }
    local_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      local_X[i][j] = global_X[local_rank*local_bound+i][j];
    }
  }

  local_nnz = 0;
  for(int i=local_rank*local_bound; i<local_rank*local_bound+local_bound; ++i) {
    for(int j=0; j<num_vertice; ++j) {
      if(global_A[i][j] != 0) {
        local_nnz += 1;
      }
    }
  }

  local_V_COO = new float[local_nnz];
  local_COL_COO = new int[local_nnz];
  local_ROW_COO = new int[local_nnz];

  int temp = 0;
  for(int i=local_rank*local_bound; i<local_rank*local_bound+local_bound; ++i) {
    for(int j=0; j<num_vertice; ++j) {
      if(global_A[i][j] != 0) {
        local_V_COO[temp] = global_A[i][j];
        local_ROW_COO[temp] = i-local_rank*local_bound;
        local_COL_COO[temp] = j;
        ++temp;
      }
    }
  }

  local_V_HiCOO = new float[local_nnz];
  local_COL_HiCOO = new char[local_nnz];
  local_ROW_HiCOO = new char[local_nnz];
  temp_local_BLK_COL_HiCOO = new int[local_nnz];
  temp_local_BLK_ROW_HiCOO = new int[local_nnz];

  blk_size = 0;
  if (local_bound < BLK_SIZE) {
    blk_size = local_bound;
  } else {
    blk_size = BLK_SIZE;
  }

  temp = 0;
  for(int i=local_rank*local_bound; i<local_rank*local_bound+local_bound; i+=blk_size) {
    for(int j=0; j<num_vertice; j+=blk_size) {
      for(int ii=i; ii<min(i+blk_size, local_rank*local_bound+local_bound); ++ii) {
        for(int jj=j; jj<min(j+blk_size, num_vertice); ++jj) {
          if(global_A[ii][jj] != 0) {
            local_V_HiCOO[temp] = global_A[ii][jj];
            local_ROW_HiCOO[temp] = char((ii-local_rank*local_bound)%blk_size);
            local_COL_HiCOO[temp] = char(jj%blk_size);
            temp_local_BLK_ROW_HiCOO[temp] = (i-local_rank*local_bound)/blk_size;
            temp_local_BLK_COL_HiCOO[temp] = j/blk_size;
            if(temp != 0 and (temp_local_BLK_COL_HiCOO[temp] != temp_local_BLK_COL_HiCOO[temp-1] or temp_local_BLK_ROW_HiCOO[temp] != temp_local_BLK_ROW_HiCOO[temp-1])) {
              local_nnb++;
            }
            ++temp;
          }
        }
      }
    }
  }

  local_BLK_COL_HiCOO = new int[local_nnb];
  local_BLK_ROW_HiCOO = new int[local_nnb];
  local_BLK_SIZE_HiCOO = new int[local_nnb];
  for(int i=0; i<local_nnb; ++i) {
    local_BLK_SIZE_HiCOO[i] = 0;
  }
  temp = 0;
  local_BLK_COL_HiCOO[temp] = temp_local_BLK_COL_HiCOO[0];
  local_BLK_ROW_HiCOO[temp] = temp_local_BLK_ROW_HiCOO[0];
  local_BLK_SIZE_HiCOO[temp] += 1;
  for(int i=1; i<local_nnz; ++i) {
    if(temp_local_BLK_COL_HiCOO[i] != temp_local_BLK_COL_HiCOO[i-1] or temp_local_BLK_ROW_HiCOO[i] != temp_local_BLK_ROW_HiCOO[i-1]) {
      temp++;
      local_BLK_COL_HiCOO[temp] = temp_local_BLK_COL_HiCOO[i];
      local_BLK_ROW_HiCOO[temp] = temp_local_BLK_ROW_HiCOO[i];
      local_BLK_SIZE_HiCOO[temp] = local_BLK_SIZE_HiCOO[temp-1] + 1;
    } else {
      local_BLK_SIZE_HiCOO[temp] += 1;
    }
  }

  communicate_buffer = new float[num_vertice];
  temp_communicate_buffer = new float[num_vertice];
  current_feature = new float[num_vertice/NODES];
  out0 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out0[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      out0[i][j] = 0;
    }
  }

  out1 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out1[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      out1[i][j] = 0;
    }
  }

  out2 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out2[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      out2[i][j] = 0;
    }
  }

  out3 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out3[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      out3[i][j] = 0;
    }
  }
}

bool verify(float** a, float** b) {
  int base = local_rank*(num_vertice/NODES);
  for(int i=local_rank*(num_vertice/NODES); i<(local_rank+1)*(num_vertice/NODES); ++i) {
    for(int j=0; j<num_w1_out; ++j) {
      if(abs(a[i-base][j] - b[i][j]) > 0.001) {
        cout<<"rank: "<<local_rank<<" local_out["<<i-base<<"]["<<j<<"]: "<<a[i-base][j]<<"; gold["<<i<<"]["<<j<<"]: "<<b[i][j]<<endl;
        return false;
      }
    }
  }
  return true;
}

void display_input() {
  if(local_rank == 0) {
    cout<<"[init A] rank "<<local_rank<<" : "<<endl;
    for(int i=0; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_vertice; ++j) {
        cout<<local_A[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
    cout<<"[init X] rank "<<local_rank<<" : "<<endl;
    for(int i=0; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_feature; ++j) {
        cout<<local_X[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[init weight0] rank "<<local_rank<<" : "<<endl;
    for(int i=0; i<num_feature; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_w0_out; ++j) {
        cout<<global_weight0[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[init bias0] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int j=0; j<num_w0_out; ++j) {
      cout<<global_bias0[j]<<" ";
    }
    cout<<" ]"<<endl;
  
    cout<<"[init weight1] rank "<<local_rank<<" : "<<endl;
    for(int i=0; i<num_w0_out; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_w1_out; ++j) {
        cout<<global_weight1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[init bias1] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int j=0; j<num_w1_out; ++j) {
      cout<<global_bias1[j]<<" ";
    }
    cout<<" ]"<<endl;
//  }

    cout<<"[init COO local_V_COO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnz; ++i) {
      cout<<local_V_COO[i]<<" ";
    }
    cout<<" ]"<<endl;

    cout<<"[init COO local_COL_COO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnz; ++i) {
      cout<<local_COL_COO[i]<<" ";
    }
    cout<<" ]"<<endl;

    cout<<"[init COO local_ROW_COO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnz; ++i) {
      cout<<local_ROW_COO[i]<<" ";
    }
    cout<<" ]"<<endl;

    cout<<"[init HiCOO local_V_HiCOO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnz; ++i) {
      cout<<local_V_HiCOO[i]<<" ";
    }
    cout<<" ]"<<endl;

    cout<<"[init HiCOO local_COL_HiCOO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnz; ++i) {
      cout<<int(local_COL_HiCOO[i])<<" ";
    }
    cout<<" ]"<<endl;

    cout<<"[init HiCOO local_ROW_HiCOO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnz; ++i) {
      cout<<int(local_ROW_HiCOO[i])<<" ";
    }
    cout<<" ]"<<endl;

    cout<<"[init HiCOO local_BLK_COL_HiCOO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnb; ++i) {
      cout<<local_BLK_COL_HiCOO[i]<<" ";
    }
    cout<<" ]"<<endl;

    cout<<"[init HiCOO local_BLK_ROW_HiCOO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnb; ++i) {
      cout<<local_BLK_ROW_HiCOO[i]<<" ";
    }
    cout<<" ]"<<endl;

    cout<<"[init HiCOO local_BLK_SIZE_HiCOO] rank "<<local_rank<<" : "<<endl;
    cout<<"[ ";
    for(int i=0; i<local_nnb; ++i) {
      cout<<local_BLK_SIZE_HiCOO[i]<<" ";
    }
    cout<<" ]"<<endl;
  }
}

void display_output() {
//  if(local_rank == NODES-1) {
    cout<<"[output out0] rank "<<local_rank<<" : "<<endl;
    for(int i=0; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_feature; ++j) {
        cout<<out0[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
    cout<<"[output out1] rank "<<local_rank<<" : "<<endl;
    for(int i=0; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_w0_out; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
//  }
}
 
// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// ----------------------------------------------------------------------
void ax0_spmv_kernel(int f) {
//  for(int i=0; i<num_vertice/NODES; ++i) {
//      for(int k=local_ROW_COO[i]; k<local_ROW_COO[i+1]; ++k) {
//        out0[i][f] += local_V_COO[k] * communicate_buffer[local_COL_COO[k]]; 
//      }
//  }

//  for(int k=0; k<local_nnz; ++k) {
//    out0[local_ROW_HiCOO[k]][f] += local_V_HiCOO[k] * communicate_buffer[local_COL_HiCOO[k]]; 
//  }

  for(int b=0; b<local_nnb; ++b) {
//    if(local_rank==0)
//    cout<<"[DEBUG] rank "<<local_rank<<" blk: "<<b<<endl;
    int start = 0;
    if(b != 0) {
      start = local_BLK_SIZE_HiCOO[b-1];
    }
    int blk_row = local_BLK_ROW_HiCOO[b]*blk_size;
    int blk_col = local_BLK_COL_HiCOO[b]*blk_size;
    for(int k=start; k<local_BLK_SIZE_HiCOO[b]; ++k) {
      int row = blk_row+int(local_ROW_HiCOO[k]);
      int col = blk_col+int(local_COL_HiCOO[k]);
//    if(local_rank==0)
//      cout<<"[DEBUG] rank "<<local_rank<<" k: "<<k<<"; row: "<<row<<"; col: "<<col<<endl;
      out0[row][f] += local_V_HiCOO[k] * communicate_buffer[col]; 
//    if(local_rank==0)
//      cout<<"[DEBUG] rank "<<local_rank<<" k: "<<k<<"; row: "<<row<<"; col: "<<col<<" done"<<endl;
    }
  }

  MPI_Test(&request_profile, &flag_profile, &status_profile);
}

float temp = 0.0;
void mw0_kernel() {
  for(int k=0; k<num_w0_out; ++k) {
    for(int i=0; i<num_vertice/NODES; ++i) {
      temp = out1[i][k];
      for(int j=0; j<num_feature; ++j) {
        temp += out0[i][j] * global_weight0[j][k];
      }
      out1[i][k] = temp + global_bias0[k];
      if(out1[i][k] < 0)
        out1[i][k] = 0;
    }
  }
}

void ax1_spmv_kernel(int f) {
//  for(int i=0; i<num_vertice/NODES; ++i) {
//    for(int k=local_ROW_COO[i]; k<local_ROW_COO[i+1]; ++k) {
//      out2[i][f] += local_V_COO[k] * communicate_buffer[local_COL_COO[k]]; 
//    }
//  }

//  for(int k=0; k<local_nnz; ++k) {
//    out2[local_ROW_HiCOO[k]][f] += local_V_HiCOO[k] * communicate_buffer[local_COL_HiCOO[k]]; 
//  }

  for(int b=0; b<local_nnb; ++b) {
    int start = 0;
    if(b != 0) {
      start = local_BLK_SIZE_HiCOO[b-1];
    }
    int blk_row = local_BLK_ROW_HiCOO[b]*blk_size;
    int blk_col = local_BLK_COL_HiCOO[b]*blk_size;
    for(int k=start; k<local_BLK_SIZE_HiCOO[b]; ++k) {
      int row = blk_row+int(local_ROW_HiCOO[k]);
      int col = blk_col+int(local_COL_HiCOO[k]);
      out2[row][f] += local_V_HiCOO[k] * communicate_buffer[col]; 
    }
  }

  MPI_Test(&request_profile, &flag_profile, &status_profile);
}

void mw1_kernel() {
  for(int k=0; k<num_w1_out; ++k) {
    for(int i=0; i<num_vertice/NODES; ++i) {
      temp = out3[i][k];
      for(int j=0; j<num_w0_out; ++j) {
        temp += out2[i][j] * global_weight1[j][k];
      }
      out3[i][k] = temp + global_bias1[k];
    }
  }
}

void init_task(int argc, char *argv[]) {
  // MPI initial
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

  // TODO: Task start point.
  global_start = 0;

  init_data();
  // display_input();
}

// ----------------------------------------------------------------------
// Main function.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // Initialize global data start and end
  init_task(argc, argv);

  if(nodes != NODES) {cout<<"NODES do not match"<<endl;return -1;}

  MPI_Barrier(MPI_COMM_WORLD);
  chrono::system_clock::time_point start = chrono::system_clock::now();

  // Execution
  // Layer 1 -- M = A x X:
  memset(temp_communicate_buffer, 0, sizeof(float)*num_vertice);
  for(int k=0; k<num_feature; ++k) {
    for(int j=0; j<num_vertice/NODES; ++j) {
      temp_communicate_buffer[local_rank*local_bound+j] = local_X[j][k];
    }
    MPI_Allreduce(temp_communicate_buffer, communicate_buffer, num_vertice, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    total_data_out += num_vertice;
    total_data_in += num_vertice;
    ax0_spmv_kernel(k);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Layer 1 -- M = M x Weight + bias
  mw0_kernel();

  // Layer 2 -- M = A x M:
  memset(temp_communicate_buffer, 0, sizeof(float)*num_vertice);
  for(int k=0; k<num_w0_out; ++k) {
    for(int j=0; j<num_vertice/NODES; ++j) {
      temp_communicate_buffer[local_rank*local_bound+j] = out1[j][k];
    }
    MPI_Allreduce(temp_communicate_buffer, communicate_buffer, num_vertice, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    total_data_out += num_vertice;
    total_data_in += num_vertice;
    ax1_spmv_kernel(k);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  // Layer 2 -- M = M x Weight + bias
  mw1_kernel();

  chrono::system_clock::time_point end = chrono::system_clock::now();

  if(verify(out3, output_gold)) {
    cout<<"success~"<<endl;
  } else {
    cout<<"fail.."<<endl;
//    display_output();
  }

  if(local_rank != NODES) {
    cout<<"[final] rank "<<local_rank<<" out0: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_feature-1; j<num_feature; ++j) {
        cout<<out0[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[final] rank "<<local_rank<<" out1: ";
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[final] rank "<<local_rank<<" out2: ";
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
        cout<<out2[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[final] rank "<<local_rank<<" out3: ";
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w1_out-1; j<num_w1_out; ++j) {
        cout<<out3[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  }

  // ------------ print out timing -------------
  chrono::duration<double> elapsed_seconds = end-start;
  cout<<"[time] rank "<<local_rank<<" elapsed time: "
      <<elapsed_seconds.count()<<"s"<<endl;
       // " finished computation at " << ctime(&end_time)
  cout<<"[data movement] rank "<<local_rank<<" data in: "<<total_data_in<<" data out: "<<total_data_out<<" total: "<<total_data_in+total_data_out<<" size: "<<4*(total_data_in+total_data_out)<<endl;

  MPI_Finalize();
  return 0;
}

