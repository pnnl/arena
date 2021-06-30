// =======================================================================
// gcn_ax.cpp
// =======================================================================
// Single layer GCN's Ax implementation, which can be viewed as GEMV.
// Note that the entire implementation of single layer GCN is FC(Ax, W).
//
// Author : Cheng Tan
//   Date : Nov 1, 2020

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
#define NODES 1

using namespace std;

int local_start;
int local_end;
int local_bound;
int global_start;

int nodes;
int local_rank = 0;

long int total_data_in  = 0;
long int total_data_out = 0;

int flag_profile;

// ----------------------------------------------------------------------
// Local data allocated onto a node.
// TODO: user specified.
// ----------------------------------------------------------------------
#define SPARSE 3
int num_vertice;
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

float** local_X;
float* communicate_buffer;
float* current_feature;
float** out0;
float** out1;
float** out2;
float** out3;

void init_data() {

#ifdef DUMMY_DATA
  num_vertice = 4;
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
  output_gold[0][0] = 338;
  output_gold[1][0] = 462;
  output_gold[2][0] = 200;
  output_gold[3][0] = 548;

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
    }
  }
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

  communicate_buffer = new float[num_vertice/NODES];
  current_feature = new float[num_vertice/NODES];
//  x = new int[num_vertice/NODES];
//  for(int i=0; i<num_vertice/NODES; ++i) {
//    x[i] = i+local_rank*local_bound;
//    for(int j=0; j<num_vertice; ++j) {
//      x[i][j] = i+j;
//    }
//  }
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
//  if(local_rank == NODES-1) {
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
// TODO: user specified.
// ----------------------------------------------------------------------
void ax0_kernel(int start, int k) {
//  int temp1 = 0;
  for(int i=0; i<num_vertice/NODES; ++i) {
//    float temp0 = out0[i][k];
    float temp0 = 0;
    float temp1 = 0;
    float temp2 = 0;
    float temp3 = 0;
// note that the vectorization version works for single core sequential
// implementation, instead of multiple-node MPI version
#ifdef VECTORIZATION
    for(int j=0; j<local_bound; j+=4) {
      temp0 += local_A[i][start+j+0] * communicate_buffer[j+0];
      temp1 += local_A[i][start+j+1] * communicate_buffer[j+1];
      temp2 += local_A[i][start+j+2] * communicate_buffer[j+2];
      temp3 += local_A[i][start+j+3] * communicate_buffer[j+3];
    }
#endif
#ifndef VECTORIZATION
    for(int j=0; j<local_bound; j++) {
      temp0 += local_A[i][start+j] * communicate_buffer[j];
//      cout<<"rank "<<local_rank<<" is processing... local_A["<<i<<"]["<<start+j<<"]: "<<local_A[i][start+j]<<" * comm["<<j<<"]: "<<communicate_buffer[j]<<endl;
    }
#endif
    out0[i][k] += temp0 + temp1 + temp2 + temp3;
#ifdef DEBUG
    cout<<"[result] rank "<<local_rank<<" out["<<i<<"]["<<k<<"]: "<<out0[i][k]<<endl;
#endif
  }
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

void ax1_kernel(int start, int k) {
  for(int i=0; i<num_vertice/NODES; ++i) {
    temp = out2[i][k];
    for(int j=0; j<local_bound; ++j) {
      temp += local_A[i][start+j] * current_feature[j];
//      cout<<"rank "<<local_rank<<" is processing... out["<<i<<"]: "<<out[i]<<" = local_A["<<i<<"]["<<start+j<<"]: "<<local_A[i][start+j]<<" * local_temp["<<j<<"]: "<<local_temp[j]<<endl;
#ifdef DEBUG
      cout<<"[result] rank "<<local_rank<<" out["<<i<<"]["<<k<<"]: "<<out0[i][k]<<endl;
#endif
    }
    out2[i][k] = temp;
  }
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
  // TODO: Task start point.
  global_start = 0;
  nodes = NODES;

  init_data();
//  display_input();
}

// ----------------------------------------------------------------------
// Main function.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // Initialize global data start and end
  init_task(argc, argv);

  if(nodes != NODES) {cout<<"NODES do not match"<<endl;return -1;}

  chrono::system_clock::time_point start = chrono::system_clock::now();

  // Execution
  // Layer 1 -- M = A x X:
  for(int k=0; k<num_feature; ++k) {
    for(int i=0; i<NODES; ++i) {
      for(int j=0; j<num_vertice/NODES; ++j) {
        communicate_buffer[j] = local_X[j][k];
      }
      if(i == local_rank) {
        total_data_out += num_vertice;
      }
      total_data_in += num_vertice/NODES;
      ax0_kernel(i*local_bound, k);
    }
  }

  // Layer 1 -- M = M x Weight + bias
  mw0_kernel();

  // Layer 2 -- M = A x M:
  for(int k=0; k<num_w0_out; ++k) {
    for(int i=0; i<NODES; ++i) {
      for(int j=0; j<num_vertice/NODES; ++j) {
        communicate_buffer[j] = out1[j][k];
      }
      for(int j=0; j<num_vertice/NODES; ++j) {
        current_feature[j] = communicate_buffer[j];
      }
      if(i == local_rank) {
        total_data_out += num_vertice;
      } else {
        total_data_in += num_vertice/NODES;
      }
      ax1_kernel(i*local_bound, k);
    }
  }
  // Layer 2 -- M = M x Weight + bias
  mw1_kernel();

  chrono::system_clock::time_point end = chrono::system_clock::now();

  if(verify(out3, output_gold)) {
    cout<<"success~"<<endl;
  } else {
    cout<<"fail.."<<endl;
    display_output();
  }

  if(local_rank == NODES-1) {
    cout<<"[final] rank "<<local_rank<<" out0: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_feature-1; j<num_feature; ++j) {
//    for(int i=0; i<num_vertice/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_feature; ++j) {
        cout<<out0[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[final] rank "<<local_rank<<" out1: ";
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
//    for(int i=0; i<num_vertice/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_w0_out; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[final] rank "<<local_rank<<" out2: ";
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
//    for(int i=0; i<num_vertice/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_w0_out; ++j) {
        cout<<out2[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  
    cout<<"[final] rank "<<local_rank<<" out3: ";
//    for(int i=0; i<num_vertice/NODES; ++i) {
//      cout<<"[ ";
//      for(int j=0; j<num_w1_out; ++j) {
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
//      for(int j=num_w1_out-1; j<num_w1_out; ++j) {
      for(int j=0; j<num_w1_out; ++j) {
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

  return 0;
}

