// =======================================================================
// nbody.h
// =======================================================================
// ARENA implementation of N-Body.
//
// Author : Cheng Tan
//   Date : May 8, 2020

#include "../../lib/ARENA.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>

//#define SIZE 1024
//#define SIZE 8
//#define NODES 2
#define NODES 4
//#define NODES 8
//#define NODES 16


// ----------------------------------------------------------------------
// Total data allocated onto nodes TODO: user specified 
// ----------------------------------------------------------------------
typedef struct {
  float x, y, z;
} m_vector;
int bodies, timeSteps;
int local_bodies;
float *masses, GravConstant;
m_vector *positions, *velocities, *accelerations;
float *positions_x, *positions_y, *positions_z;
float *velocities_x, *velocities_y, *velocities_z;

m_vector addVectors(m_vector a,m_vector b) {
  m_vector c = {a.x+b.x,a.y+b.y,a.z+b.z};
  return c;
}

m_vector scaleVector(float b,m_vector a) {
  m_vector c = {b*a.x,b*a.y,b*a.z};
  return c;
}

m_vector subtractVectors(m_vector a,m_vector b) {
  m_vector c = {a.x-b.x,a.y-b.y,a.z-b.z};
  return c;
}

float mod(m_vector a) {
  return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

void initiateSystem(char* fileName) {
  int i;
  FILE* fp = fopen(fileName,"r");

  fscanf(fp,"%f%d%d",&GravConstant,&bodies,&timeSteps);

  ARENA_local_bound = ARENA_local_rank * (bodies/NODES);
  ARENA_local_start = ARENA_local_rank * (bodies/NODES);
  ARENA_local_end   = ARENA_local_rank * (bodies/NODES) + bodies/NODES;
  local_bodies = bodies/NODES;

  // TODO: Task start point.
  ARENA_global_start = 0;
  ARENA_global_end   = local_bodies;
  ARENA_global_param = 0;

  masses = (float*)malloc(bodies*sizeof(float));
  positions = (m_vector*)malloc(bodies*sizeof(m_vector));
  positions_x = (float*)malloc(bodies*sizeof(float));
  positions_y = (float*)malloc(bodies*sizeof(float));
  positions_z = (float*)malloc(bodies*sizeof(float));
  velocities = (m_vector*)malloc(bodies*sizeof(m_vector));
  velocities_x = (float*)malloc(bodies*sizeof(float));
  velocities_y = (float*)malloc(bodies*sizeof(float));
  velocities_z = (float*)malloc(bodies*sizeof(float));
  accelerations = (m_vector*)malloc(bodies*sizeof(m_vector));

  float temp = 0;

  for(i=0;i<bodies;i++) {
    fscanf(fp,"%f",&masses[i]);
    if(i<ARENA_local_start or i>=ARENA_local_end) {
      fscanf(fp,"%f%f%f",&temp,&temp,&temp);
      fscanf(fp,"%f%f%f",&temp,&temp,&temp);
    } else {
      // specific for current rank
      fscanf(fp,"%f%f%f",&positions[i].x,&positions[i].y,&positions[i].z);
      fscanf(fp,"%f%f%f",&velocities[i].x,&velocities[i].y,&velocities[i].z);
//      cout<<"[input] rank "<<ARENA_local_rank<<" finished reading pos "<<i<<" xyz: "<<positions[i].x<<" "<<positions[i].y<<" "<<positions[i].z<<endl;
    }
  }
  fclose(fp);
}

void init_data() {}
void init_kernel() {}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc) TODO: user specified
// ----------------------------------------------------------------------
int total_execution = 0;
int ARENA_kernel(int start, int end, int param) {
  int i = total_execution;
  int current_start = start;
  if(current_start < total_execution)
    current_start = total_execution;
  for(int j=current_start; j<end; j++) {
    if(positions[i].x==positions[j].x && positions[i].y==positions[j].y && positions[i].z==positions[j].z) {
      m_vector temp = velocities[i];
      velocities[i] = velocities[j];
      velocities[j] = temp;
    }
  }
    
  total_execution++;
//    cout<<"rank: "<<local_rank<<" real local_acc["<<i<<"]: "<<accelerations[i].x<<endl;
  int num_spawn = 0;
  if(ARENA_local_rank != ARENA_nodes-1) {
    ARENA_spawn[num_spawn].id         = ARENA_NORMAL_TASK;
    ARENA_spawn[num_spawn].start      = ARENA_local_end;
    ARENA_spawn[num_spawn].end        = ARENA_local_end+local_bodies;
    ARENA_spawn[num_spawn].param      = 0;
    ARENA_spawn[num_spawn].more_from  = ARENA_local_rank;
    ARENA_spawn[num_spawn].more_start = 0;
    ARENA_spawn[num_spawn].more_end   = 6;
    // Same as more_start and more_end but need indicate destination (rank+1 for lu)
    ARENA_remote_ask_start[(ARENA_local_rank+1)%NODES] = 0;
    ARENA_remote_ask_end[(ARENA_local_rank+1)%NODES] = 6;
    num_spawn += 1;
//    cout<<"[spawn remote task] rank "<<ARENA_local_rank<<": "<<total_execution-1<<endl;
  }
  if(total_execution >= ARENA_local_start and total_execution < ARENA_local_end) {
    ARENA_spawn[num_spawn].id         = ARENA_NORMAL_TASK;
    ARENA_spawn[num_spawn].start      = ARENA_local_start;
    ARENA_spawn[num_spawn].end        = ARENA_local_end;
    ARENA_spawn[num_spawn].param      = 0;
//    ARENA_spawn[num_spawn].more_from  = ARENA_local_rank;
//    ARENA_spawn[num_spawn].more_start = 0;
//    ARENA_spawn[num_spawn].more_end   = 6;
    // Same as more_start and more_end but need indicate destination (rank+1 for lu)
//    ARENA_remote_ask_start[(ARENA_local_rank+1)%NODES] = 0;
//    ARENA_remote_ask_end[(ARENA_local_rank+1)%NODES] = 6;
    num_spawn += 1;
//    cout<<"[spawn local task] rank "<<ARENA_local_rank<<": "<<total_execution-1<<endl;
  }

//  cout<<"[OUT] rank "<<ARENA_local_rank<<": ";
//  for(int x=0; x<SIZE/NODES; ++x) {
//    for(int y=0; y<SIZE; ++y) {
//      cout<<" "<<local_OUT[x][y];
//    }
//    cout<<" || ";
//  }
//  cout<<endl;
  return num_spawn;
}

void ARENA_init_task(int argc, char *argv[], int nodes) {

  // MPI initial
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ARENA_nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //ARENA_nodes = nodes;
  ARENA_local_rank = rank;

  if(argc!=2)
    printf("Usage : %s <file name containing system configuration data>",argv[0]);
  else{
    initiateSystem(argv[1]);
  }

  init_data();

  // TODO: Remote data requirement. The second parameter indicates
  //       wheter the data depends on the predecessor task
  ARENA_init_data_buff(1, true);
  ARENA_remote_ask_buff[(rank+1)%NODES] = new float[6];
  ARENA_local_need_buff[(NODES+rank-1)%NODES] = new float[6];
}

int main(int argc, char *argv[]) {

  // Initialize global data start and end
  ARENA_init_task(argc, argv, NODES);

  // Initialize local allocated data
  init_kernel();

  // Execute kernel
  ARENA_run();

  // Output
  cout<<"[output] rank "<<ARENA_local_rank<<endl;
//  for(int j=ARENA_local_start; j<ARENA_local_end; ++j) {
//    cout<<"x y z: "<<accelerations[j].x<<" "<<accelerations[j].y<<" "<<accelerations[j].z<<endl;
//  }

  return 0;
}

// ----------------------------------------------------------------------
// Prepare data to send to remote nodes.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
int send_data_times = 0;
void ARENA_load_data(int start, int end, float* buff) {
  buff[0] =  positions[send_data_times].x;
  buff[1] =  positions[send_data_times].y;
  buff[2] =  positions[send_data_times].z;
  buff[3] = velocities[send_data_times].x;
  buff[4] = velocities[send_data_times].y;
  buff[5] = velocities[send_data_times].z;
//  cout<<"[SEE SENT DATA] rank "<<ARENA_local_rank<<": ";
//  for(int i=start;i<end; ++i) {
//    cout<<buff[i]<<" ";
//  }
//  cout<<endl;
  send_data_times++;
  return;
}

// ----------------------------------------------------------------------
// Receive data from remote nodes and store into local memory.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
int recv_data_times = 0;
void ARENA_store_data(int start, int end, int source, float* buff) {
  positions[recv_data_times].x = buff[0];
  positions[recv_data_times].y = buff[1];
  positions[recv_data_times].z = buff[2];
  velocities[recv_data_times].x = buff[3];
  velocities[recv_data_times].y = buff[4];
  velocities[recv_data_times].z = buff[5];
  //cout<<"[SEE RECEIVED DATA] rank "<<ARENA_local_rank<<": ";
  //for(int i=0;i<local_bodies; ++i) {
  //  cout<<buff[i*3+0]<<" "<<buff[i*3+1]<<" "<<buff[i*3+2];
  //}
  //cout<<endl;
  recv_data_times++;
  return;
}

