// ======================================================================
// gcn_2_layer.cpp
// ======================================================================
// ARENA implementation of 2 layer GCN for CORA dataset.
// The adjacent Matrix is reprensented in COO format. 
//
// Mechanism: Sparse matrix multiplication on its local data then stream
//            the data to the next location indicated by start/end.
//            Theoratically, it has the same amount of data movement as
//            the conventional bulk-synchronization MPI (broadcast-based)
//            solution.
//
// Benefit:   The computaton and communication can be asynchronous
//            compared to the bulk-synchronization solution.
//
// Problem:   Tried to think about it in a push-based style, but the task
//            delivery would probably dominate the communication (i.e.,
//            data transfer are translated into task transfer in the
//            context of ARENA). So still make it pull-based (then stream
//            the local features).
//
// TODO: 1. Vector -> Matrix.
//       2. Load CORA 2-layer weights.
//       3. Invoke second layer workload by spawning new tasks.
//
// Author : Cheng Tan
//   Date : Jan 5, 2021

//#define DEBUG 
#include "../../../../lib/ARENA.h"
#include <iostream>
#include <fstream>
#include <string>

//#define DUMMY_DATA

#define KERNEL_LAYER0 2
#define KERNEL_LAYER0_ACCUM 3
#define KERNEL_LAYER1 4
#define KERNEL_LAYER1_ACCUM 5

//#define NODES 2
#define NODES 4
//#define NODES 8
//#define NODES 16

#define BLK_SIZE 20

int local_rank;
int local_start;
int local_end;

int num_vertice;
int num_feature;
int num_w0_out;
int num_w1_out;
int** global_A;
int** local_A;
float** global_X;
float** local_X;
float** global_weight0;
float* global_bias0;
float** global_weight1;
float* global_bias1;
float* buff_X;
float** recv_X;
float** trans_X;
float** out0;
float** out1;
float** trans_out1;
float** out2;
float** out3;
float** temp_out;
float** output_gold;
int* offset0;
int* offset1;
bool* first_round_store;
int LAYER0_OPT_ALL;
int LAYER1_OPT_ALL;
void display_input();          // helper function
void display_output();         // helper function
bool verify(float**, float**); // helper function

int num_nonzero_A = 0;         // for csr format
int local_nnz;                 // for csr format
int local_nnb = 1;
int blk_size;
float* local_V;                // for csr format
int* local_COL;                // for csr format
int* local_ROW;                // for csr format

float* local_V_HiCOO;
int* temp_local_BLK_COL_HiCOO;
int* temp_local_BLK_ROW_HiCOO;

char* local_COL_HiCOO;
int* local_BLK_COL_HiCOO;
int* local_BLK_ROW_HiCOO;
int* local_BLK_SIZE_HiCOO;
int range;

int* data_send_times;
int* data_recv_times;

bool** sent_tag;

// ----------------------------------------------------------------------
// local data initialization.
// TODO: user specified.
// ----------------------------------------------------------------------
void init_local_data();
void output();

void reset_sent_tag() {
  for(int i=0; i<num_vertice/NODES; ++i) {
    for(int j=0; j<NODES; ++j) {
      sent_tag[i][j] = false;
    }
  }
}

void mw0_kernel() {
  float temp = 0;
  for(int k=0; k<num_w0_out; ++k) {
    for(int i=0; i<num_vertice/NODES; ++i) {
      temp = out1[i][k];
      for(int j=0; j<num_feature; ++j) {
        temp += out0[i][j] * global_weight0[j][k];
      }
      out1[i][k] = temp + global_bias0[k];
      if(out1[i][k] < 0)
        out1[i][k] = 0;
      trans_out1[k][i] = out1[i][k];
    }
  }
}

void mw1_kernel() {
  float temp = 0;
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

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc).
// Note that there are three params and one return.
// TODO: user specified.
// ----------------------------------------------------------------------
//int total_times = 0;
int opt_count = 0;
int global_start = 0;
void ARENA_kernel0(long long int start, long long int end, int param, bool require_data, int length) {

  // iterate across the value inside a specific row
  int begin = 0;
  if(param != 0) {
    begin = local_BLK_SIZE_HiCOO[param-1];
  }
  int row = local_BLK_ROW_HiCOO[param];
  int blk_col = local_BLK_COL_HiCOO[param]*blk_size;
  for(int i=begin; i<local_BLK_SIZE_HiCOO[param]; ++i) {
    int col = blk_col + local_COL_HiCOO[i];
    // if the index is inside my own data range, accumulate it locally
    if(col >= ARENA_local_start and col < ARENA_local_end) {
      for(int x=0; x<num_feature; ++x) {
        out0[col-ARENA_local_start][x] += local_X[row][x];
      }
      // count the number of operation to guide the start of next layer
      ++opt_count;
    } else {
      // NOTE that ARENA does not allow send to one destination multiple times at one shot! Thus, we only send features to a destination once.
      int dest = col/range;
      if(!sent_tag[row][dest]) {
        sent_tag[row][dest] = true;
        ARENA_spawn_task(KERNEL_LAYER0_ACCUM, col, col+1, 0,
                         local_X[row], num_feature);
        global_start += num_feature;
      } else {
        ARENA_spawn_task(KERNEL_LAYER0_ACCUM, col, col+1, 0);
      }
    }
  }
  
  // iterating the rows by spawning new local tasks untill the boundary
  if(param < local_nnb-1) {
    ARENA_spawn_task(KERNEL_LAYER0, ARENA_local_start, ARENA_local_end, param+1);
  }

  // start next layer
  if(opt_count == local_nnz) {
    opt_count = 0;
    mw0_kernel();
    reset_sent_tag();
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start, ARENA_local_end, 0);
  }
}

void ARENA_kernel0_accum(long long int start, long long int end, int param, bool require_data, int length) {
  ++opt_count;

  for(int i=0; i<num_feature; ++i) {
    out0[start][i] += ARENA_recv_data_buffer[i];
  }

  // start next layer
  if(opt_count == local_nnz) {
    opt_count = 0;
    mw0_kernel();
    reset_sent_tag();
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start, ARENA_local_end, 0);
  }
}

int cur_layer = 0;
int local_cur_index1 = 0;
int k_dim;
void ARENA_kernel1(long long int start, long long int end, int param, bool require_data, int length) {
  cur_layer = 1;

  // iterate across the value inside a specific row
  int begin = 0;
  if(param != 0) {
    begin = local_BLK_SIZE_HiCOO[param-1];
  }
  int row = local_BLK_ROW_HiCOO[param];
  int blk_col = local_BLK_COL_HiCOO[param]*blk_size;
  for(int i=begin; i<local_BLK_SIZE_HiCOO[param]; ++i) {
    int col = blk_col + local_COL_HiCOO[i];
    // if the index is inside my own data range, accumulate it locally
    if(col >= ARENA_local_start and col < ARENA_local_end) {
      for(int x=0; x<num_w0_out; ++x) {
        out2[col-ARENA_local_start][x] += out1[row][x];
      }
      // count the number of operation to guide the start of next layer
      ++opt_count;
    } else {
      // NOTE that ARENA does not allow send to one destination multiple times at one shot! Thus, we only send features to a destination once.
      int dest = col/range;
      if(!sent_tag[row][dest]) {
        sent_tag[row][dest] = true;
        ARENA_spawn_task(KERNEL_LAYER1_ACCUM, col, col+1, 0,
                         out1[row], num_w0_out);
        global_start += num_w0_out;
        // ARENA_remote_ask_start[col/range].push(0);
        // ARENA_remote_ask_end[col/range].push(num_w0_out);
      } else {
        ARENA_spawn_task(KERNEL_LAYER1_ACCUM, col, col+1, 0);
      }
    }
  }
  
  // iterating the rows by spawning new local tasks untill the boundary
  if(param < local_nnb-1) {
    ARENA_spawn_task(KERNEL_LAYER1, ARENA_local_start, ARENA_local_end, param+1);
  }

  // start next layer
  if(opt_count == local_nnz) {
    opt_count = 0; 
    mw1_kernel();
  }

}

void ARENA_kernel1_accum(long long int start, long long int end, int param, bool require_data, int length) {
  ++opt_count;
  for(int i=0; i<num_w0_out; ++i) {
    out2[start][i] += ARENA_recv_data_buffer[i];
  }

  // start next layer
  if(opt_count == local_nnz) {
    opt_count = 0;
    mw1_kernel();
  }
}

// ----------------------------------------------------------------------
// Main function. No need to change.
// ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // Initialize global data start and end
  local_rank = ARENA_init(NODES);

  init_local_data();

  local_start = local_rank * (num_vertice/NODES);
  local_end   = local_rank * (num_vertice/NODES) + (num_vertice/NODES);
  ARENA_set_local(local_start, local_end);

  // Set task start point
  long long int root_start = 0;
  long long int root_end   = num_vertice;
  int root_param = 0;

  ARENA_register_task(KERNEL_LAYER0, &ARENA_kernel0, true, root_start, root_end, root_param);
  ARENA_register_task(KERNEL_LAYER0_ACCUM, &ARENA_kernel0_accum);
  ARENA_register_task(KERNEL_LAYER1, &ARENA_kernel1);
  ARENA_register_task(KERNEL_LAYER1_ACCUM, &ARENA_kernel1_accum);

  // Display local allocated data
  // display_input();

  // Execute kernel
  ARENA_run();

  // Verify
  if(verify(out3, output_gold)) {
    cout<<"rank "<<ARENA_local_rank<<" success~"<<endl;
  } else {
    cout<<"rank "<<ARENA_local_rank<<" fail.."<<endl;
    display_output();
  }

  output();
 
  return 0;
}

// ======================================================================
// helper functions (e.g., print out input/output, verify results)
// ----------------------------------------------------------------------
void display_input() {
  if (ARENA_local_rank != -1) { 
  cout<<"[init A] rank "<<ARENA_local_rank<<" : "<<num_vertice/NODES<<": "<<endl;
  for(int i=0; i<num_vertice/NODES; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_vertice; ++j) {
      cout<<local_A[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }
  cout<<"[init X] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<num_vertice/NODES; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_feature; ++j) {
      cout<<local_X[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }

  cout<<"[init recv_X] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<num_vertice; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_feature; ++j) {
      cout<<recv_X[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }

  cout<<"[init weight1] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<num_feature; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_w0_out; ++j) {
      cout<<global_weight0[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }

  cout<<"[init bias1] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int j=0; j<num_w0_out; ++j) {
    cout<<global_bias0[j]<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init weight2] rank "<<ARENA_local_rank<<" : "<<endl;
  for(int i=0; i<num_w0_out; ++i) {
    cout<<"[ ";
    for(int j=0; j<num_w1_out; ++j) {
      cout<<global_weight1[i][j]<<" ";
    }
    cout<<" ]"<<endl;
  }

  cout<<"[init bias2] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int j=0; j<num_w1_out; ++j) {
    cout<<global_bias1[j]<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init HiCOO local_V_HiCOO] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int i=0; i<local_nnz; ++i) {
    cout<<local_V_HiCOO[i]<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init HiCOO local_COL_HiCOO] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int i=0; i<local_nnz; ++i) {
    cout<<int(local_COL_HiCOO[i])<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init HiCOO local_BLK_COL_HiCOO] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int i=0; i<local_nnb; ++i) {
    cout<<local_BLK_COL_HiCOO[i]<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init HiCOO local_BLK_ROW_HiCOO] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int i=0; i<local_nnb; ++i) {
    cout<<local_BLK_ROW_HiCOO[i]<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init HiCOO local_BLK_SIZE_HiCOO] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int i=0; i<local_nnb; ++i) {
    cout<<local_BLK_SIZE_HiCOO[i]<<" ";
  }
  cout<<" ]"<<endl;

  cout<<"[init data_send_times] rank "<<ARENA_local_rank<<" : "<<endl;
  cout<<"[ ";
  for(int i=0; i<num_vertice/NODES; ++i) {
    cout<<data_send_times[i]<<" ";
  }
  cout<<" ]"<<endl;

  }
}

void display_output() {
//  if(local_rank == NODES-1) {
    cout<<"[output out0] rank "<<ARENA_local_rank<<" : "<<endl;
    for(int i=0; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_feature; ++j) {
        cout<<out0[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
    cout<<"[output out1] rank "<<ARENA_local_rank<<" : "<<endl;
    for(int i=0; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=0; j<num_w0_out; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
//  }
}

bool verify(float** a, float** b) {
  int base = ARENA_local_rank*(num_vertice/NODES);
  for(int i=ARENA_local_rank*(num_vertice/NODES); i<(ARENA_local_rank+1)*(num_vertice/NODES); ++i) {
    for(int j=0; j<num_w1_out; ++j) {
      if(abs(a[i-base][j] - b[i][j]) > 0.001) {
        cout<<"rank: "<<ARENA_local_rank<<" local_out["<<i-base<<"]["<<j<<"]: "<<a[i-base][j]<<"; gold["<<i<<"]["<<j<<"]: "<<b[i][j]<<endl;
        return false;
      }
    }
  }
  return true;
}

void init_local_data() {
#ifdef DUMMY_DATA
  num_vertice = 8;
  num_feature = 4;
  num_w0_out = 2;
  num_w1_out = 1;

  global_A = new int*[num_vertice];
  global_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_A[i] = new int[num_vertice];
    for(int j=0; j<num_vertice; ++j) {
      global_A[i][j] = 0;//(i+j)%2;//(i*num_vertice+j);
    }
  }
  for(int i=0; i<num_vertice; ++i) {
    for(int j=i; j<num_vertice; ++j) {
      global_A[i][j] = j%2;
      global_A[j][i] = j%2;
    }
  }

  global_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    global_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      //global_X[i][j] = i + j;
      global_X[i][j] = i*num_feature + j;
    }
  }

  global_weight0 = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    global_weight0[i] = new float[num_w0_out];
    for(int j=0; j<num_w0_out; ++j) {
      global_weight0[i][j] = i;//i%3+j%3;///(num_feature*1.0);
    }
  }

  global_bias0 = new float[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    global_bias0[i] = 0;//i%2;///(num_feature*1.0);
  }

  global_weight1 = new float*[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    global_weight1[i] = new float[num_w1_out];
    for(int j=0; j<num_w1_out; ++j) {
      global_weight1[i][j] = i*num_w1_out+j;//i%3+j%3;
    }
  }

  global_bias1 = new float[num_w1_out];
  for(int i=0; i<num_w1_out; ++i) {
    global_bias1[i] = 0;//(i+1)%2;///(num_w1_out*1.0);
  }

  output_gold = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    output_gold[i] = new float[num_w1_out];
  }
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

  data_send_times = new int[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    data_send_times[i] = 0;
  }
  data_recv_times = new int[NODES];
  for(int i=0; i<NODES; ++i) {
    data_recv_times[i] = 0;
  }

  for(int i=ARENA_local_rank*num_vertice/NODES; i<(ARENA_local_rank+1)*num_vertice/NODES; ++i) {
    for(int j=0; j<num_vertice; j+=num_vertice/NODES) {
      for(int jj=j; jj<min(num_vertice, j+num_vertice/NODES); ++jj) {
        if(global_A[i][jj] != 0 and (jj<ARENA_local_rank*num_vertice/NODES or jj>=(ARENA_local_rank+1)*num_vertice/NODES)) {
          data_send_times[i-ARENA_local_rank*num_vertice/NODES]++;
          break;
        }
      }
    }
  }

  for(int i=0; i<num_vertice; ++i) {
    for(int j=0; j<num_vertice; ++j) {
      if(global_A[i][j] != 0) {
        num_nonzero_A += 1;
      }
    }
  }

  local_nnz = 0;
  for(int i=ARENA_local_rank*num_vertice/NODES; i<(ARENA_local_rank+1)*num_vertice/NODES; ++i) {
    for(int j=0; j<num_vertice; ++j) {
      if(global_A[i][j] != 0) {
        local_nnz += 1;
      }
    }
  }
  local_V = new float[local_nnz];
  local_COL = new int[local_nnz];
  local_ROW = new int[local_nnz];

  int temp = 0;
  for(int i=ARENA_local_rank*num_vertice/NODES; i<(ARENA_local_rank+1)*num_vertice/NODES; ++i) {
    for(int j=0; j<num_vertice; ++j) {
      if(global_A[i][j] != 0) {
        local_V[temp] = global_A[i][j];
        local_ROW[temp] = i-ARENA_local_rank*num_vertice/NODES;
        local_COL[temp] = j;
        ++temp;
      }
    }
  }

  local_V_HiCOO = new float[local_nnz];
  local_COL_HiCOO = new char[local_nnz];
  temp_local_BLK_COL_HiCOO = new int[local_nnz];
  temp_local_BLK_ROW_HiCOO = new int[local_nnz];

  range = num_vertice/NODES;
  blk_size = 0;
  if (range< BLK_SIZE) {
    blk_size = range;
  } else {
    blk_size = BLK_SIZE;
  }

  temp = 0;
  for(int i=ARENA_local_rank*range; i<ARENA_local_rank*range+range; ++i) {
    for(int j=0; j<num_vertice; j+=blk_size) {
      for(int jj=j; jj<min(j+blk_size, num_vertice); ++jj) {
        if(global_A[i][jj] != 0) {
          local_V_HiCOO[temp] = global_A[i][jj];
          local_COL_HiCOO[temp] = char(jj%blk_size);
          temp_local_BLK_ROW_HiCOO[temp] = i-ARENA_local_rank*range;
          temp_local_BLK_COL_HiCOO[temp] = j/blk_size;
          if(temp != 0 and (temp_local_BLK_COL_HiCOO[temp] != temp_local_BLK_COL_HiCOO[temp-1] or temp_local_BLK_ROW_HiCOO[temp] != temp_local_BLK_ROW_HiCOO[temp-1])) {
            local_nnb++;
          }
          ++temp;
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

  sent_tag = new bool*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    sent_tag[i] = new bool[NODES];
    for(int j=0; j<NODES; ++j) {
      sent_tag[i][j] = false;
    }
  }

  LAYER0_OPT_ALL = NODES*num_feature;
  LAYER1_OPT_ALL = NODES*num_w0_out;

  local_A = new int*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    local_A[i] = new int[num_vertice];
    for(int j=0; j<num_vertice; ++j) {
      local_A[i][j] = global_A[ARENA_local_rank*(num_vertice/NODES)+i][j];
    }
  }

  local_X = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    local_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      local_X[i][j] = global_X[ARENA_local_rank*(num_vertice/NODES)+i][j];
    }
  }

  int local_start = ARENA_local_rank*(num_vertice/NODES);
  int local_end = (ARENA_local_rank+1)*(num_vertice/NODES);

  buff_X = new float[num_feature];
  recv_X = new float*[num_vertice];
  for(int i=0; i<num_vertice; ++i) {
    recv_X[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      recv_X[i][j] = 0;
      if(i >= local_start and i < local_end) {
        recv_X[i][j] = local_X[i-local_start][j];
      }
    }
  }

  trans_X = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    trans_X[i] = new float[num_vertice/NODES];
    for(int j=0; j<num_vertice/NODES; ++j) {
      trans_X[i][j] = local_X[j][i];
    }
  }

  trans_out1 = new float*[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    trans_out1[i] = new float[num_vertice/NODES];
    for(int j=0; j<num_vertice/NODES; ++j) {
      trans_out1[i][j] = 0;
    }
  }

  out0 = new float*[num_vertice/NODES];
  for(int i=0; i<num_vertice/NODES; ++i) {
    out0[i] = new float[num_feature];
    for(int j=0; j<num_feature; ++j) {
      out0[i][j] = 0;
    }
  }
  temp_out = new float*[num_feature];
  for(int i=0; i<num_feature; ++i) {
    temp_out[i] = new float[num_vertice/NODES];
    for(int j=0; j<num_vertice/NODES; ++j) {
//      temp_out[j] = 0;
      temp_out[i][j] = 0;
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

  // offset indicates the offset of the current region of the global nodes
  offset0 = new int[num_feature];
  for(int i=0; i<num_feature; ++i) {
    offset0[i] = 0;
  }

  offset1 = new int[num_w0_out];
  for(int i=0; i<num_w0_out; ++i) {
    offset1[i] = 0;
  }

  first_round_store= new bool[num_feature];
  for(int i=0; i<num_feature; ++i) {
    first_round_store[i] = false;
  }
}

void output() {
  // Output
  if(ARENA_local_rank == 0) {
    cout<<"[final] rank "<<ARENA_local_rank<<" out0: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_feature-1; j<num_feature; ++j) {
        cout<<"out0["<<i<<"]["<<j<<"]: "<<out0[i][j]<<"; ";
      }
      cout<<" ]"<<endl;
    }
    cout<<"[final] rank "<<ARENA_local_rank<<" out1: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
        cout<<out1[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

//    cout<<"[final] rank "<<ARENA_local_rank<<" trans_out1: "<<endl;
//    for(int i=num_w0_out-1; i<num_w0_out; ++i) {
//      cout<<"[ ";
//      for(int j=num_vertice/NODES-1; j<num_vertice/NODES; ++j) {
//        cout<<trans_out1[i][j]<<" ";
//      }
//      cout<<" ]"<<endl;
//    }

    cout<<"[final] rank "<<ARENA_local_rank<<" out2: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w0_out-1; j<num_w0_out; ++j) {
        cout<<out2[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }

    cout<<"[final] rank "<<ARENA_local_rank<<" out3: "<<endl;
    for(int i=num_vertice/NODES-1; i<num_vertice/NODES; ++i) {
      cout<<"[ ";
      for(int j=num_w1_out-1; j<num_w1_out; ++j) {
        cout<<out3[i][j]<<" ";
      }
      cout<<" ]"<<endl;
    }
  }

}
