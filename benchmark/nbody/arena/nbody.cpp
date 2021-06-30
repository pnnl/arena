// =======================================================================
// nbody.h
// =======================================================================
// ARENA implementation of N-Body.
//
// Author : Cheng Tan
//   Date : May 8, 2020

#include "../../../lib/ARENA.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>

//#define SIZE 1024
//#define SIZE 8
#define NODES 4
//#define NODES 8
//#define NODES 16

int local_rank;
int local_start;
int local_end;

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
float *positions_buff;

int send_data_times = 0;
int recv_data_times = 0;

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

  local_bodies = bodies/NODES;

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


  positions_buff = (float*)malloc(3*local_bodies*sizeof(float));

  local_start = local_rank * (bodies/NODES);
  local_end   = local_rank * (bodies/NODES) + bodies/NODES;

  float temp = 0;
  for(i=0;i<bodies;i++) {
    fscanf(fp,"%f",&masses[i]);
    if(i<local_start or i>=local_end) {
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

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc) TODO: user specified
// ----------------------------------------------------------------------
int total_execution = 0;
void ARENA_kernel(long long int start, long long int end, int param, bool require_data, int length) {
//  if(total_execution == -1) {
//    total_execution = ARENA_local_start;
//  }

  if(require_data) {
    for(int i=0; i<local_bodies; ++i) {
      positions[(bodies+local_start-(1+recv_data_times)*local_bodies)%bodies+i].x = ARENA_recv_data_buffer[i*3 + 0];
      positions[(bodies+local_start-(1+recv_data_times)*local_bodies)%bodies+i].y = ARENA_recv_data_buffer[i*3 + 1];
      positions[(bodies+local_start-(1+recv_data_times)*local_bodies)%bodies+i].z = ARENA_recv_data_buffer[i*3 + 2];
    }
    recv_data_times++;
  }


  if(total_execution < bodies/local_bodies) {
    int j = 0;
    for(int x=local_start;x<local_end;x++) {
      j = (bodies-total_execution*local_bodies+x)%bodies;
      if(total_execution == 0 and x == local_start) {
        for(int i=local_start; i<local_end; i++) {
          accelerations[i].x = 0;
          accelerations[i].y = 0;
          accelerations[i].z = 0;
        }
      }
      for(int i=local_start; i<local_end; i++) {
        if(i!=j) {
  //        cout<<"rank: "<<local_rank<<" is executing... param: "<<param<<"; position "<<j<<" xyz: "<<positions[j].x<<" "<<positions[j].y<<" "<<positions[j].z<<"; position "<<i<<" xyz: "<<positions[i].x<<" "<<positions[i].y<<" "<<positions[i].z<<endl;
          accelerations[i] = addVectors(accelerations[i], scaleVector(GravConstant*masses[j]/pow(mod(subtractVectors(positions[i], positions[j])), 3), subtractVectors(positions[j], positions[i])));
        }
  //      cout<<"rank: "<<ARENA_local_rank<<" [detail in "<<i<<" body "<<j<<"] x y z: "<<accelerations[i].x<<" "<<accelerations[i].y<<" "<<accelerations[i].z<<endl;
      }
    }
    if(total_execution == bodies/local_bodies-1) {
      for(int i=local_start; i<local_end; i++) {
        positions[i] = addVectors(positions[i], addVectors(velocities[i], scaleVector(0.5, accelerations[i])));
      }
      for(int i=local_start; i<local_end; i++) {
        velocities[i] = addVectors(velocities[i],accelerations[i]);
      }
//      if(ARENA_local_rank == 0) {
//        int i = 0;
//        for(j=i+1; j<ARENA_local_end; j++) {
//          if(positions[i].x==positions[j].x && positions[i].y==positions[j].y && positions[i].z==positions[j].z) {
//            m_vector temp = velocities[i];
//            velocities[i] = velocities[j];
//            velocities[j] = temp;
//          }
//        }
//      }
    }
  } else {
//    if(ARENA_local_rank == 0 and total_execution==bodies/local_bodies) {
//    for(i=local_bound; i<upper_bound; i++) {
////    for(j=i+1; j<bodies; j++) {
//      for(j=i+1; j<local_bound+local_bodies; j++) {
//
//        if(positions[i].x==positions[j].x && positions[i].y==positions[j].y && positions[i].z==positions[j].z) {
//          m_vector temp = velocities[i];
//          velocities[i] = velocities[j];
//          velocities[j] = temp;
//        }
//      }
//    }
  }

  total_execution++;
//    cout<<"rank: "<<local_rank<<" real local_acc["<<i<<"]: "<<accelerations[i].x<<endl;

//  int num_spawn = 0;
  if(total_execution*local_bodies < bodies) {
//    ARENA_spawn[num_spawn].id         = ARENA_NORMAL_TASK;
//    ARENA_spawn[num_spawn].start      = ARENA_local_end%bodies;
//    ARENA_spawn[num_spawn].end        = ARENA_local_end%bodies+local_bodies;
//    ARENA_spawn[num_spawn].param      = 0;
//    ARENA_spawn[num_spawn].more_from  = ARENA_local_rank;
//    ARENA_spawn[num_spawn].more_start = 0;
//    ARENA_spawn[num_spawn].more_end   = 3 * local_bodies;
//    // Same as more_start and more_end but need indicate destination (rank+1 for lu)
//    ARENA_remote_ask_start[(ARENA_local_rank+1)%NODES] = 0;
//    ARENA_remote_ask_end[(ARENA_local_rank+1)%NODES] = 3 * local_bodies;
//    num_spawn += 1;

    for(int i=0; i<local_bodies; ++i) {
      positions_buff[i*3 + 0] = positions[(bodies+local_start-send_data_times*local_bodies)%bodies+i].x;
      positions_buff[i*3 + 1] = positions[(bodies+local_start-send_data_times*local_bodies)%bodies+i].y;
      positions_buff[i*3 + 2] = positions[(bodies+local_start-send_data_times*local_bodies)%bodies+i].z;
    }
    send_data_times++;
    ARENA_spawn_task(ARENA_NORMAL_TASK, local_end%bodies, 
                     local_end%bodies + local_bodies, 0, &positions_buff[0],
                     3 * local_bodies);

//  } else if(total_execution >= bodies/local_bodies) {
//    if(ARENA_local_rank < NODES-1) {
//      ARENA_spawn[num_spawn].id         = ARENA_NORMAL_TASK;
//      ARENA_spawn[num_spawn].start      = ARENA_local_end;
//      ARENA_spawn[num_spawn].end        = ARENA_local_end + ARENA_local_bodies;
//      ARENA_spawn[num_spawn].param      = 0;
//      ARENA_spawn[num_spawn].more_from  = ARENA_local_rank;
//      ARENA_spawn[num_spawn].more_start = 0;
//      ARENA_spawn[num_spawn].more_end   = 6;
//      // Same as more_start and more_end but need indicate destination (rank+1 for lu)
//      ARENA_remote_ask_start[(ARENA_local_rank+1)%NODES] = 0;
//      ARENA_remote_ask_end[(ARENA_local_rank+1)%NODES] = 6;
//      num_spawn += 1;
//    }
//    ARENA_spawn[num_spawn].id         = ARENA_NORMAL_TASK;
//    ARENA_spawn[num_spawn].start      = ARENA_local_start;
//    ARENA_spawn[num_spawn].end        = ARENA_local_end;
//    ARENA_spawn[num_spawn].param      = 0;
//    ARENA_spawn[num_spawn].more_from  = -1;
//    ARENA_spawn[num_spawn].more_start = 0;
//    ARENA_spawn[num_spawn].more_end   = 0;
//    num_spawn += 1;
  }
//  cout<<"[OUT] rank "<<ARENA_local_rank<<": ";
//  for(int x=0; x<SIZE/NODES; ++x) {
//    for(int y=0; y<SIZE; ++y) {
//      cout<<" "<<local_OUT[x][y];
//    }
//    cout<<" || ";
//  }
//  cout<<endl;
//  return num_spawn;
}

void init_local_data(int argc, char *argv[]) {

  if(argc!=2)
    printf("Usage : %s <file name containing system configuration data>",argv[0]);
  else{
    initiateSystem(argv[1]);
  }
}

int main(int argc, char *argv[]) {

  // Initialize global data start and end
  local_rank = ARENA_init(NODES);

  init_local_data(argc, argv);

  ARENA_set_local(local_start, local_end);

  // Register kernel
  long long int root_start = 0;
  long long int root_end = bodies;
  int root_param = 0;
  ARENA_register_task(ARENA_NORMAL_TASK, &ARENA_kernel, true, root_start, root_end, root_param);

  // Execute kernel
  ARENA_run();

  // Output
//  if(local_rank == 0) {
//      for(int j=local_start; j<local_start+local_bodies; j++)
//        printf("Body %d : %f\t%f\t%f\t|\t%f\t%f\t%f (rank: %d)\n",j+1,positions[j].x,positions[j].y,positions[j].z,velocities[j].x,velocities[j].y,velocities[j].z, local_rank);
//  }

  return 0;
}

/*
// ----------------------------------------------------------------------
// Prepare data to send to remote nodes.
// TODO: user specified if necessary
// ----------------------------------------------------------------------
void ARENA_load_data(int start, int end, float* buff) {
  for(int i=0; i<local_bodies; ++i) {
    buff[i*3 + 0] = positions[(bodies+local_start-send_data_times*local_bodies)%bodies+i].x;
    buff[i*3 + 1] = positions[(bodies+local_start-send_data_times*local_bodies)%bodies+i].y;
    buff[i*3 + 2] = positions[(bodies+local_start-send_data_times*local_bodies)%bodies+i].z;
  }
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
  for(int i=0; i<local_bodies; ++i) {
    positions[(bodies+NA_local_start-(1+recv_data_times)*local_bodies)%bodies+i].x = buff[i*3 + 0];
    positions[(bodies+ARENA_local_start-(1+recv_data_times)*local_bodies)%bodies+i].y = buff[i*3 + 1];
    positions[(bodies+ARENA_local_start-(1+recv_data_times)*local_bodies)%bodies+i].z = buff[i*3 + 2];
  }

  //cout<<"[SEE RECEIVED DATA] rank "<<ARENA_local_rank<<": ";
  //for(int i=0;i<local_bodies; ++i) {
  //  cout<<buff[i*3+0]<<" "<<buff[i*3+1]<<" "<<buff[i*3+2];
  //}
  //cout<<endl;
  recv_data_times++;
  return;
}
*/
