// =======================================================================
// nbody.cpp
// =======================================================================
// Conventional task-centric N-body MPI implementation.
//
// Author : Cheng Tan
//   Date : May 6, 2020

#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <string>

//#define SIZE 16384
#define NODES 4

using namespace std;

long int total_data_in  = 0;
long int total_data_out = 0;

int local_start;
int local_end;
int local_bound;
int global_start;
int nodes;
int local_rank;

int flag_profile;
MPI_Request request_profile = MPI_REQUEST_NULL;
MPI_Status  status_profile;

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

  local_bound = local_rank * (bodies/NODES);
  local_start = local_rank * (bodies/NODES);
  local_end   = local_rank * (bodies/NODES) + bodies/NODES;
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

  float temp = 0;
  
  for(i=0;i<bodies;i++) {
    fscanf(fp,"%f",&masses[i]);
    if(i<local_bound or i>=local_bound+local_bodies) {
      fscanf(fp,"%f%f%f",&temp,&temp,&temp);
      fscanf(fp,"%f%f%f",&temp,&temp,&temp);
    } else {
      // specific for current rank
//      if(local_rank == 0)
//        printf("local_rank: %d, position[%d].x=%f\n", local_rank, i, positions[i].x);
      fscanf(fp,"%f%f%f",&positions[i].x,&positions[i].y,&positions[i].z);
      fscanf(fp,"%f%f%f",&velocities[i].x,&velocities[i].y,&velocities[i].z);
    }
  }
  fclose(fp);
}
 
void resolveCollisions() {
  // need broadcast velocities here.
  int i,j;
 
  int upper_bound = local_bound+local_bodies;
  if(local_rank == nodes - 1) {
    upper_bound -= 1;
  }

  if(local_rank != 0) {
    for(i=0; i<local_bound; ++i) {
      MPI_Recv(positions_x+i, 1, MPI_FLOAT, local_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(positions_y+i, 1, MPI_FLOAT, local_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(positions_z+i, 1, MPI_FLOAT, local_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(velocities_x+i, 1, MPI_FLOAT, local_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(velocities_y+i, 1, MPI_FLOAT, local_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(velocities_z+i, 1, MPI_FLOAT, local_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      total_data_in += 6;
      positions[i].x = positions_x[i];
      positions[i].y = positions_y[i];
      positions[i].z = positions_z[i];
      velocities[i].x = velocities_x[i];
      velocities[i].y = velocities_y[i];
      velocities[i].z = velocities_z[i];

      for(j=local_bound; j<local_bound+local_bodies; j++) {
        
        if(positions[i].x==positions[j].x && positions[i].y==positions[j].y && positions[i].z==positions[j].z) {
          m_vector temp = velocities[i];
          velocities[i] = velocities[j];
          velocities[j] = temp;
        }
      }

      if(local_rank!=nodes-1) {
        positions_x[i] = positions[i].x;
        positions_y[i] = positions[i].y;
        positions_z[i] = positions[i].z;
        velocities_x[i] = velocities[i].x;
        velocities_y[i] = velocities[i].y;
        velocities_z[i] = velocities[i].z;
       
        MPI_Send(positions_x+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
        MPI_Send(positions_y+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
        MPI_Send(positions_z+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
        MPI_Send(velocities_x+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
        MPI_Send(velocities_y+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
        MPI_Send(velocities_z+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
        total_data_out += 6;
      }
    }
  }

  for(i=local_bound; i<upper_bound; i++) {
//    for(j=i+1; j<bodies; j++) {
    for(j=i+1; j<local_bound+local_bodies; j++) {
      
      if(positions[i].x==positions[j].x && positions[i].y==positions[j].y && positions[i].z==positions[j].z) {
        m_vector temp = velocities[i];
        velocities[i] = velocities[j];
        velocities[j] = temp;
      }
    }
    if(local_rank!=nodes-1) {
      positions_x[i]  = positions[i].x;
      positions_y[i]  = positions[i].y;
      positions_z[i]  = positions[i].z;
      velocities_x[i] = velocities[i].x;
      velocities_y[i] = velocities[i].y;
      velocities_z[i] = velocities[i].z;
     
      MPI_Send(positions_x +i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
      MPI_Send(positions_y +i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
      MPI_Send(positions_z +i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
      MPI_Send(velocities_x+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
      MPI_Send(velocities_y+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
      MPI_Send(velocities_z+i, 1, MPI_FLOAT, local_rank+1, 0, MPI_COMM_WORLD);
      total_data_out += 6;
    }
  }
  if(local_rank==nodes-1) {
    for(i=0; i<local_bound; ++i) {
      velocities_x[i] = velocities[i].x;
      velocities_y[i] = velocities[i].y;
      velocities_z[i] = velocities[i].z;
     
      MPI_Send(velocities_x+i, 1, MPI_FLOAT, i/local_bodies, 0, MPI_COMM_WORLD);
      MPI_Send(velocities_y+i, 1, MPI_FLOAT, i/local_bodies, 0, MPI_COMM_WORLD);
      MPI_Send(velocities_z+i, 1, MPI_FLOAT, i/local_bodies, 0, MPI_COMM_WORLD);
      total_data_out += 3;
    }
  } else {
    for(i=local_bound; i<local_bound+local_bodies; ++i) {
      velocities_x[i] = velocities[i].x;
      velocities_y[i] = velocities[i].y;
      velocities_z[i] = velocities[i].z;
     
      MPI_Recv(velocities_x+i, 1, MPI_FLOAT, nodes-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(velocities_y+i, 1, MPI_FLOAT, nodes-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(velocities_z+i, 1, MPI_FLOAT, nodes-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      total_data_in += 3;
      velocities[i].x = velocities_x[i];
      velocities[i].y = velocities_y[i];
      velocities[i].z = velocities_z[i];
    }
  }
}
 
void computeAccelerations() {
  // need all gather positions here.
  int i,j;

  for(i=local_bound; i<local_bound+local_bodies; i++) {
    accelerations[i].x = 0;
    accelerations[i].y = 0;
    accelerations[i].z = 0;
      
    for(j=0;j<bodies;j++) {
      if(j%local_bodies==0 and i==local_bound) {
        if(local_rank == j/local_bodies) {
          for(int x=local_bound; x<local_bound+local_bodies; x++) {
            positions_x[x] = positions[x].x;
            positions_y[x] = positions[x].y;
            positions_z[x] = positions[x].z;
          }
        }
        MPI_Bcast(positions_x+local_bodies*(j/local_bodies), local_bodies, MPI_FLOAT,
                  j/local_bodies, MPI_COMM_WORLD);
        MPI_Bcast(positions_y+local_bodies*(j/local_bodies), local_bodies, MPI_FLOAT,
                  j/local_bodies, MPI_COMM_WORLD);
        MPI_Bcast(positions_z+local_bodies*(j/local_bodies), local_bodies, MPI_FLOAT,
                  j/local_bodies, MPI_COMM_WORLD);
        if(local_rank == j/local_bodies) {
          total_data_out += NODES*3*local_bodies;
        } else {
          total_data_in += 3*local_bodies;
        }
        if(local_rank != j/local_bodies) {
          for(int x=local_bodies*(j/local_bodies); x<local_bodies*(j/local_bodies)+local_bodies; x++) {
            positions[x].x = positions_x[x];
            positions[x].y = positions_y[x];
            positions[x].z = positions_z[x];
          }
        }
      }

      if(i!=j) {
        accelerations[i] = addVectors(accelerations[i], scaleVector(GravConstant*masses[j]/pow(mod(subtractVectors(positions[i], positions[j])), 3), subtractVectors(positions[j], positions[i])));
      }
    }
//    cout<<"rank: "<<local_rank<<" real local_acc["<<i<<"]: "<<accelerations[i].x<<endl;
  }
}
 
void computeVelocities() {
  int i;
  for(i=local_bound; i<local_bound+local_bodies; i++) {
    velocities[i] = addVectors(velocities[i],accelerations[i]);
//    cout<<"rank: "<<local_rank<<" real local_vel["<<i<<"]: "<<velocities[i].x<<endl;
  }
}
 
void computePositions() {
  int i;
  for(i=local_bound; i<local_bound+local_bodies; i++) {
    positions[i] = addVectors(positions[i], addVectors(velocities[i], scaleVector(0.5, accelerations[i])));
//    cout<<"rank: "<<local_rank<<" real local_pos["<<i<<"]: "<<positions[i].x<<endl;
  }
}
 
void simulate() {
  computeAccelerations();
//  computePositions();
//  computeVelocities();
//  resolveCollisions();
}
 
int main(int argc,char* argv[]) {

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

  int i,j;
  
  chrono::system_clock::time_point start;
  if(argc!=2)
    printf("Usage : %s <file name containing system configuration data>",argv[0]);
  else{
    initiateSystem(argv[1]);
    // TODO: Data tag.
//    printf("Body   :     x              y               z           |           vx              vy              vz   (rank: %d)", local_rank);
    start = chrono::system_clock::now();
    for(i=0; i<timeSteps; i++) {
//      printf("\nCycle %d\n",i+1);
      simulate();
//      for(j=local_bound; j<local_bound+local_bodies; j++)
//        printf("Body %d : %f\t%f\t%f\t|\t%f\t%f\t%f (rank: %d)\n",j+1,positions[j].x,positions[j].y,positions[j].z,velocities[j].x,velocities[j].y,velocities[j].z, local_rank);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  chrono::system_clock::time_point end = chrono::system_clock::now();

  MPI_Finalize();
  chrono::duration<float> elapsed_seconds = end-start;
  time_t end_time = chrono::system_clock::to_time_t(end);
  cout<<"[time] rank "<<local_rank<<" elapsed time: "
      <<elapsed_seconds.count()<<"s"<<endl;

  cout<<"[data movement] rank "<<local_rank<<" data in: "<<total_data_in<<" data out: "<<total_data_out<<" total: "<<total_data_in+total_data_out<<" size: "<<4*(total_data_in+total_data_out)<<endl;


//  // OUTPUT
//  if(local_rank == 0) {
//      for(j=local_bound; j<local_bound+local_bodies; j++)
//        printf("Body %d : %f\t%f\t%f\t|\t%f\t%f\t%f (rank: %d)\n",j+1,positions[j].x,positions[j].y,positions[j].z,velocities[j].x,velocities[j].y,velocities[j].z, local_rank);
//  }
  return 0;
}

