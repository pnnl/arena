#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>

using namespace std;

#define NUM_THREADS 3

int world_rank;

void *PrintHello(void *threadid) {
   long tid;
   tid = (long)threadid;
   cout << "Process "<<world_rank<<". Hello World! Thread ID " << tid << endl;
   pthread_exit(NULL);
}

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // We are assuming at least 2 processes for this task
  if (world_size < 2) {
    fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int number;
  if (world_rank == 0) {
    // If we are rank 0, set the number to -1 and send it to process 1
    number = -1;
    MPI_Send(
      /* data         = */ &number, 
      /* count        = */ 1, 
      /* datatype     = */ MPI_INT, 
      /* destination  = */ 1, 
      /* tag          = */ 0, 
      /* communicator = */ MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(
      /* data         = */ &number, 
      /* count        = */ 1, 
      /* datatype     = */ MPI_INT, 
      /* source       = */ 0, 
      /* tag          = */ 0, 
      /* communicator = */ MPI_COMM_WORLD, 
      /* status       = */ MPI_STATUS_IGNORE);
    cout<<"Process 1 received number "<<number<<" from process 0"<<endl;

    pthread_t threads[NUM_THREADS];
    int rc;
    int i;
    
    for( i = 0; i < NUM_THREADS; i++ ) {
       rc = pthread_create(&threads[i], NULL, PrintHello, (void *)i);
       
       if (rc) {
          cout << "Error:unable to create thread," << rc << endl;
          exit(-1);
       }
    }
    pthread_exit(NULL);
  }
  MPI_Finalize();
}
