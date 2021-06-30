#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<iostream>
#include <chrono>
#include <ctime>
 
using namespace std;

typedef struct{
  float x,y,z;
}m_vector;
 
int bodies,timeSteps;
float *masses,GravConstant;
m_vector *positions,*velocities,*accelerations;
 
m_vector addVectors(m_vector a,m_vector b){
  m_vector c = {a.x+b.x,a.y+b.y,a.z+b.z};
  return c;
}
 
m_vector scaleVector(float b,m_vector a){
  m_vector c = {b*a.x,b*a.y,b*a.z};
  return c;
}
 
m_vector subtractVectors(m_vector a,m_vector b){
  m_vector c = {a.x-b.x,a.y-b.y,a.z-b.z};
  return c;
}
 
float mod(m_vector a){
  return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}
 
void initiateSystem(char* fileName){
  int i;
  FILE* fp = fopen(fileName,"r");
  
  fscanf(fp,"%f%d%d",&GravConstant,&bodies,&timeSteps);
  
  masses = (float*)malloc(bodies*sizeof(float));
  positions = (m_vector*)malloc(bodies*sizeof(m_vector));
  velocities = (m_vector*)malloc(bodies*sizeof(m_vector));
  accelerations = (m_vector*)malloc(bodies*sizeof(m_vector));
  
  for(i=0;i<bodies;i++){
    fscanf(fp,"%f",&masses[i]);
    fscanf(fp,"%f%f%f",&positions[i].x,&positions[i].y,&positions[i].z);
    fscanf(fp,"%f%f%f",&velocities[i].x,&velocities[i].y,&velocities[i].z);
  }
  
  fclose(fp);
}
 
void resolveCollisions(){
  int i,j;
  for(i=0;i<bodies-1;i++)
    for(j=i+1;j<bodies;j++){
      if(positions[i].x==positions[j].x &&
         positions[i].y==positions[j].y &&
         positions[i].z==positions[j].z){
        m_vector temp = velocities[i];
        velocities[i] = velocities[j];
        velocities[j] = temp;
      }
    }
}
 
void computeAccelerations(){
  int i,j;
  for(i=0;i<bodies;i++){
    accelerations[i].x = 0;
    accelerations[i].y = 0;
    accelerations[i].z = 0;
    for(j=0;j<bodies;j++){
      if(i!=j){
      	accelerations[i] = addVectors(accelerations[i],scaleVector(GravConstant*masses[j]/pow(mod(subtractVectors(positions[i],positions[j])),3),subtractVectors(positions[j],positions[i])));
      }
    }
  }
}
 
void computeVelocities(){
  int i;
  
  for(i=0;i<bodies;i++)
    velocities[i] = addVectors(velocities[i],accelerations[i]);
}
 
void computePositions(){
  int i;
  
  for(i=0;i<bodies;i++)
    positions[i] = addVectors(positions[i],addVectors(velocities[i],scaleVector(0.5,accelerations[i])));
}
 
void simulate(){
  computeAccelerations();
  computePositions();
  computeVelocities();
  resolveCollisions();
}
 
int main(int argC,char* argV[]) {
  int i,j;
  
  chrono::system_clock::time_point start;
  chrono::system_clock::time_point end;
  if(argC!=2)
    printf("Usage : %s <file name containing system configuration data>",argV[0]);
  else{
    initiateSystem(argV[1]);
//    printf("Body   :     x              y               z           |           vx              vy              vz   ");
    for(i=0;i<timeSteps;i++){
//      printf("\nCycle %d\n",i+1);
      start = chrono::system_clock::now();
      simulate();
      end = chrono::system_clock::now();
//      for(j=0;j<bodies;j++)
//        printf("Body %d : %f\t%f\t%f\t|\t%f\t%f\t%f\n",j+1,positions[j].x,positions[j].y,positions[j].z,velocities[j].x,velocities[j].y,velocities[j].z);
    }
  }
  chrono::duration<float> elapsed_seconds = end-start;
  time_t end_time = chrono::system_clock::to_time_t(end);
  cout<<"[time] elapsed time: "
      <<elapsed_seconds.count()<<"s"<<endl;
  return 0;
}
