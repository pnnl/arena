Getting Start with ARENA
=======

ARENA is a Computing Flow Programming Model that can be enable in a distributed system with CPU and MPI supported. The programming model include ARENA runtime (ARENA.h), ARENA-template, ARENA-sim which are used to create, run and estimate computing flow applications. In this tutorial, you will work with helloworld example to learn how to create and run you first computing flow application.

## Preparing Environment ##
To run the following example, you need to install and setup MPI and C++ environment. We tested ARENA using the following MPI and C++ version:
- gcc version gcc/7.1 or later
- MPI version OpenMPI-3.1.5 

## Running HelloWorld ##

![Example](https://github.com/tancheng/arena/blob/master/tutorial/pics/pingpang.PNG)

For our "Hello, World" example with ARENA, we use the classical PingPang problem: using two MPI-threads to update a counter: one for counter increasing and the other for printing the latest number. The two threads also print their local string "Hello," and "World!". To show the computing flow execution, one thread depends the other one to spawn tasks and the bpcounter will be sent via task spawning.    

The source code for the example is located in 
```
./tutorial/hello/arena/helloworld.cpp
```

### Compile ###
To compile the example, move to the .cpp direction and simply typy `sh compile.sh`. This step generates binary file `a.out` for running.

### Run ###
Run the example by specifying two nodes you used by typing:
```
mpirun -np 2 a.out
```
or simply type `sh run.sh`.

By running the example, you may get the following output:
```
[init] rank 0
rank 0 Init done!Hello,
[init] rank 1
rank 1 Init done!World!
Hello, World! 1
Hello, World! 2
Hello, World! 3
Hello, World! 4
Hello, World! 5
Hello, World! 6
Hello, World! 7
Hello, World! 8
Hello, World! 9
Hello, World! 10
[time] rank 1 elapsed time: 0.196835s
[data movement] rank 1 data in: 10 data out: 9 data total: 19 total data size: 76 task in: 22 task out: 21 task total: 43 total task size: 430
[time] rank 0 elapsed time: 0.194701s
[data movement] rank 0 data in: 9 data out: 10 data total: 19 total data size: 76 task in: 20 task out: 22 task total: 42 total task size: 420

```
Please note that you may not get the "hello, world" in one line due to the differet I/O buffer timing between two MPI ranks. The result also shows the execution time and data movement in two ranks.

## Creating Computing Flow Application with ARENA ##
In this tutorial, we will use the same "Hello, World!" example to introduce how to create your own application with ARENA. 

### Import the ARENA Runtime ###
The execution of computing flow application is managed by ARENA runtime which is developed as c++ code. It also provides programming interface for your C++ computing flow application. We first include the ARENA runtime header.
```C++
#include "../../lib/ARENA.h"
```

### Define nodes and Local Data ###
Next, we need to initial and allocate the local data to each worker. In this example, we play PingPang and print "Hello, World!" in two MPI threads so we initial the local value `counter = 0` at begining in both threads. The following is the initial_kernel function in our example.

```C++
#define NODES 2
int counter;
void init_kernel( ){
  counter =0;
}

```

### Develop the Task Kernel ###
Instead of transmit data, ARENA programming model spawns task kernel to remote or local computing resource via ARENA runtime and ARENA API. So the next step to design computing flow application is develop the task kernel that run in target computing resource.
The following is the implementation of HelloWorld:
```C++
int HelloWorld(int start, int end, int param, bool require_data, int length) {
    if (local_start == 0){
    cout << "hello, ";
    // counter increment
    assert(counter == param);
    counter++;
    if(counter < 10) {
        // spawn task and deliver it in a ping-pong manner
        ARENA_spawn_task(HELLO_TASK, 21, 25, counter);
    }
  } else if (local_start  == 20){
    // counter update
    counter = param;
    cout << "World! "<< counter << endl;
    if(counter < 10) {
        // spawn task and deliver it in a ping-pong manner
        ARENA_spawn_task(HELLO_TASK, 0, 5, counter);
    }
  }
}

```
- HelloWorld is the user defined task name, you can used any name you liked.
- start and end identify the local data range which will be set by ARENA runtime based on the required data in the kernel. The range also be used by ARENA runtime to split and allocate the task to target computing resource.
- param is the task parameter, the value is set during task spawn.
- ARENA_local_start(end) is the position of allocated computing resource.
- root_start(end is needed) points to the global data address.
- ARENA_spawn_task(int task_ID, int global_start, int global_end, float param, float* data_start, int data_length) is the ARENA API to spwan task to the network and send data to remote rank. We also show how to the spawn "HelloWorld" without using data transmission above. 
- data_start and data_length indicate the data will be needed by the remote node.

During the task running, the task display the "Hello," or "World!" based on the global data start. After that, "Hello" will increase the counter while "World" print the counter. This task also spawn a new task if `counter<10`.

### Main Function ###
The following is the main function of our example.
```C++
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
```
1. ARENA Initial
   - Same as normal MPI application, we need to first initial the rank ID to get `local_rank`.
2. Initial root task start point
   - We setup `root_start` and `root_end` to identify the data range where root task will start and end. In this example, the root task have start point 0 and end point 1.
3. Initial local rank data range
   - We use `ARENA_set_local` to identify the data range the rank will process.
4. Register tasks/kernels
   - We using `ARENA_register(int task_ID, &Kernel, bool isFirstLaunch, int root_start, int root_end, float root_param)` to regist our kernel to ARENA runtime. To support multiple tasks, you can `#define` your own ARENA task IDs to replace `ARENA_NORMAL_TASK`. Please avoid using -1, -2.... as task ID, which have be reserved by ARENA runtime.

### Put Them All Together ###
The source code of the example can be found in [here](https://github.com/tancheng/benchmark-cfa/blob/simulator/hello/arena/helloworld.cpp).







