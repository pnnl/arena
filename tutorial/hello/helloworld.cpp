// =====================================================================
// helloworld.cpp
// =====================================================================
// ARENA implementation of helloworld
//
// Author : Chenhao Xie, Cheng Tan
//   Date : March 18, 2021

#include "../../lib/ARENA.h"
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define NODES 2

#define HELLO_TASK 1

using namespace std;

// ----------------------------------------------------------------------
// ARENA local variables. Can be customized by the user.
// ----------------------------------------------------------------------
int local_rank;
int local_start;
int local_end;
    
// ----------------------------------------------------------------------
// Local data allocated onto each node.
// ----------------------------------------------------------------------
int counter;

// ----------------------------------------------------------------------
// Initialize random local data value for the demo.
// ----------------------------------------------------------------------
void init_data( ) {
  cout<<"[init] rank "<<ARENA_local_rank<<endl;
  counter = 0;  
}

// ----------------------------------------------------------------------
// Target computation (e.g., spmv, bfs, etc)
// user specified
// [start, end)
// ----------------------------------------------------------------------
void HelloWorld(int start, int end, int param, bool require_data, int length) {
  
  if (local_start == 0){
    cout << "hello, ";
    // TODO: counter increment
    assert(counter == param);
    counter++;

    if(counter < 10) {
        // TODO: spawn task and deliver it in a ping-pong manner
        ARENA_spawn_task(HELLO_TASK, 21, 25, counter);
    }

  } else if (local_start  == 20){

    // TODO: counter update
    counter = param;

    cout << "World! "<< counter << endl;

    if(counter < 10) {
        // TODO: spawn task and deliver it in a ping-pong manner
        ARENA_spawn_task(HELLO_TASK, 0, 5, counter);
    }
  }
}


// ----------------------------------------------------------------------
// // Main function. No need to change.
// // ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

  // Initialize global data start and end
  local_rank = ARENA_init(NODES);
  if(local_rank == 0) {
    local_start = 0;
    local_end = 10;
  } else if(local_rank == 1) {
    local_start = 20;
    local_end = 30;
  }
  ARENA_set_local(local_start, local_end);

  // Register kernel
  bool isFirstLaunch = true;
  int root_start = 0;
  int root_end   = 1;
  int root_param = 0;
  ARENA_register_task(HELLO_TASK, &HelloWorld, isFirstLaunch, root_start, root_end, root_param);

  // Initialize local allocated data
  init_data();

  // Execute kernel
  ARENA_run();

  return 0;
}


