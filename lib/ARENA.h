// =======================================================================
// ARENA.h
// =======================================================================
// Header file of programming model for ARENA system architecture.
// This is a MPI-based baseline running on CPU.
//
// Author : Cheng Tan
//   Date : March 18, 2020
//
//   @xiec add tracing function

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <queue>
#include <chrono>
#include <ctime>

//#define DEBUG
#define TIME
//#define DATAMOVEMENT

#define ARENA_NO_TASK        -1
#define ARENA_TERMINATE_TASK -2
#define ARENA_NORMAL_TASK    -3
#define ARENA_TAG_SIZE        7
#define ARENA_TERMINATE_TRH   1
#define ARENA_TAG_TASK        0
#define ARENA_TAG_START       1
#define ARENA_TAG_END         2
#define ARENA_TAG_PARAM       3
#define ARENA_TAG_MORE_FROM   4
#define ARENA_TAG_MORE_START  5
#define ARENA_TAG_MORE_LENGTH 6
#define ARENA_CONTINUE        0
#define ARENA_TERMINATE       1
#define ARENA_SPAWN_MAX       4096
#define DATA_BUFF_SIZE        60000000

using namespace std;

// -----------------------------------------------------------------------
// ARENA task struc
// -----------------------------------------------------------------------
int           ARENA_nodes;
int           ARENA_root_task_id = -1;
int           ARENA_local_rank;
long long int ARENA_local_start;
long long int ARENA_local_end;
long long int ARENA_target_start;
long long int ARENA_target_end;
int           ARENA_target_id;
int           ARENA_target_param;
int           ARENA_target_more_from;
int           ARENA_target_more_start;
int           ARENA_target_more_length;
int           ARENA_remote_start;
int           ARENA_remote_end;
int           ARENA_tag[ARENA_TAG_SIZE];
long long int ARENA_root_start;
long long int ARENA_root_end;
int           ARENA_root_param;
int           ARENA_num_spawn = 0;
float**       ARENA_local_need_buff;
float*        ARENA_recv_data_buffer;
bool          ARENA_encounter_terminator = false;
bool          ARENA_sent_task            = false;
int           ARENA_terminate_count      = ARENA_TERMINATE_TRH;
bool          ARENA_has_data_delivery    = false;
bool          ARENA_data_depend_task     = false;
bool          ARENA_skip_recv            = false;
long int      ARENA_total_data_in        = 0;
long int      ARENA_total_data_out       = 0;
long int      ARENA_total_task_in        = 0;
long int      ARENA_total_task_out       = 0;
map<int, void (*)(long long int, long long int, int, bool, int)> ARENA_kernel_map;

float* window_buffer;
struct  ARENA_tag_struct {
  int id;
  long long int start;
  long long int end;
  int param;
  int more_from;
  int more_start;
  int more_length;
  ARENA_tag_struct(){}
  ARENA_tag_struct(int* tag)
    : id(tag[0]),  start(tag[1]),
      end(tag[2]), param(tag[3]),
      more_from(tag[4]),
      more_start(tag[5]),
      more_length(tag[6]){}
  ARENA_tag_struct(int id,  long long int start,
                   long long int end, int param,
                   int more_from,
                   int more_start,
                   int more_length)
    : id(id), start(start), end(end), param(param),
      more_from(more_from), more_start(more_start),
      more_length(more_length){}
};
queue<ARENA_tag_struct> ARENA_recv_list;
queue<ARENA_tag_struct> ARENA_send_list;
queue<ARENA_tag_struct> ARENA_spawn_list;
ARENA_tag_struct* ARENA_spawn;

// ARENA helper functions
inline int  ARENA_init(int, char*, int);
inline int  ARENA_task_arrive();
inline void ARENA_task_exec();
inline void ARENA_init_param();
inline void ARENA_init_data_buff();
inline void ARENA_task_analyze();
inline void ARENA_task_dispatch();
inline void ARENA_task_issue();
inline void ARENA_data_value_prepare(int, int);
inline void ARENA_data_value_receive();
inline void ARENA_fill_tag(ARENA_tag_struct);
inline void ARENA_fill_terminate_tag();
inline void ARENA_send_task_tag();

// MPI constants
//MPI_Comm    comm_world_data_value;
//MPI_Comm    comm_world_data_count;
//MPI_Group   world_group;
MPI_Status  status;
MPI_Request request_task       = MPI_REQUEST_NULL;
MPI_Request request_data_count = MPI_REQUEST_NULL;
MPI_Request request_data_value = MPI_REQUEST_NULL;

MPI_Win window;

int flag_profile = 0;
MPI_Status  status_profile;
MPI_Request request_profile = MPI_REQUEST_NULL;

//#define TRACE
#define RECV 0
#define SEND 1
#define EXEC 2
#define SPAW 3
#define DATA 0
#define TASK 1

//trace function
string trace_filename = "arena.out";
int ARENA_task_latency = 100;
int ARENA_task_size = 32;

void traceout(int local_rank, int isSend, bool isTask,int taskcycle, int remote_rank, int size){     

      string SorR, TorD;
      if (isTask) TorD = "Task";
      else TorD = "Data";
      switch(isSend){
      case SEND: {
        SorR = "Send";
        break;
        }
       case RECV: {
        SorR = "Recv";
        break;
        }
       case EXEC: {
        SorR = "Exec";
        break;
        }
       case SPAW: {
        SorR = "Spaw";
        break;
        } 
        
      }
      
      fstream fout;
      string filename = trace_filename;
      filename.append(".");
      filename.append(to_string(local_rank));
      fout.open(filename, ios::out | ios::app);
      fout << local_rank << " " << SorR << " " << TorD <<  " " << taskcycle <<  " " <<remote_rank << " " << size << "\n";
      
      fout.close();
}

inline int ARENA_init(int nodes) {
  // MPI initial
  int rank;
  //MPI_Init(&argc, &argv);
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &ARENA_nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //ARENA_nodes = nodes;
  ARENA_local_rank = rank;
  return rank;
}

// =======================================================================
// ARENA programming model busy waiting kernel.
// =======================================================================
inline int ARENA_run() {

  // Init param
  ARENA_init_param();

  bool initial = true;

#ifdef TIME
  chrono::system_clock::time_point start = chrono::system_clock::now();
#endif


  while(1) {

 //   if(ARENA_local_rank == 0 and initial)
 //     // Root task no need to receive
 //     initial = false;
 //   else {
      // Task arrival
    if(ARENA_task_arrive() == ARENA_TERMINATE) {
      cout<<"[terminate] rank "<<ARENA_local_rank<<endl;
      break;
    }
//    }
    // Analyzing arriving task
    ARENA_task_analyze();

    // Task dispatching and remote task enqueue
    ARENA_task_dispatch();

//    if(!ARENA_data_depend_task)
//      // Data send
//      ARENA_data_value_prepare();

    // Data receive if necessary
    ARENA_data_value_receive();

    // Task exec
    ARENA_task_exec();

//    if(ARENA_data_depend_task)
//      // Data send
//      ARENA_data_value_prepare();

    // Issue the spawned tasks if necessary
    ARENA_task_issue();
  }

#ifdef TIME
  chrono::system_clock::time_point end = chrono::system_clock::now();
#endif

  MPI_Win_fence(0, window);

  MPI_Finalize();

#ifdef TIME
  chrono::duration<double> elapsed_seconds = end-start;
  time_t end_time = chrono::system_clock::to_time_t(end);
  cout<<"[time] rank "<<ARENA_local_rank<<" elapsed time: "
      <<elapsed_seconds.count()<<"s"<<endl;
#endif

#ifdef DATAMOVEMENT
  cout<<"[data movement] rank "<<ARENA_local_rank<<" data in: "<<ARENA_total_data_in<<" data out: "<<ARENA_total_data_out<<" data total: "<<ARENA_total_data_in+ARENA_total_data_out<<" total data size: "<<4*(ARENA_total_data_in+ARENA_total_data_out)<<" task in: "<<ARENA_total_task_in<<" task out: "<<ARENA_total_task_out<<" task total: "<<ARENA_total_task_in+ARENA_total_task_out<<" total task size: "<<10*(ARENA_total_task_in+ARENA_total_task_out)<<endl;
#endif

//  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

// -----------------------------------------------------------------------
// Register ARENA param.
// -----------------------------------------------------------------------
inline void ARENA_register_task(int t_TAG, void (*t_kernel)(long long int, long long int, int, bool, int), 
                           bool t_root=false, int t_start = 0, int t_end = 0,
                           int t_param = -1) {
  if(t_root) {
    ARENA_root_task_id = t_TAG;
    ARENA_root_start   = t_start;
    ARENA_root_end     = t_end;
    ARENA_root_param   = t_param;
  }
  ARENA_kernel_map[t_TAG] = t_kernel;
}

inline void ARENA_set_local(long long int t_start, long long int t_end) {
  ARENA_local_start = t_start;
  ARENA_local_end   = t_end;
}

// -----------------------------------------------------------------------
// Init ARENA param.
// -----------------------------------------------------------------------
inline void ARENA_init_param() {

  // Init ARENA tag and spawn
  ARENA_spawn = new ARENA_tag_struct[ARENA_SPAWN_MAX];
  for(int i=0; i<ARENA_SPAWN_MAX; ++i) {
    ARENA_spawn[i].id = ARENA_NO_TASK;
  }

  // Init MPI
//  MPI_Comm_dup( MPI_COMM_WORLD, &comm_world_data_value );
 // MPI_Comm_group( comm_world_data_count, &world_group );
 // MPI_Comm_create_group( comm_world_data_count, world_group, 0, &comm_world_data_value );

  // Init root task
  if(ARENA_local_rank == 0) {

    ARENA_tag[ARENA_TAG_TASK]        = ARENA_TERMINATE_TASK;
    ARENA_tag[ARENA_TAG_PARAM]       = -1;
    ARENA_tag[ARENA_TAG_START]       = -1;
    ARENA_tag[ARENA_TAG_END]         = -1;
    ARENA_tag[ARENA_TAG_MORE_FROM]   = -1;
    ARENA_tag[ARENA_TAG_MORE_START]  = -1;
    ARENA_tag[ARENA_TAG_MORE_LENGTH] = -1;
    ARENA_tag_struct terminate_tag(ARENA_tag);
    ARENA_recv_list.push(terminate_tag);

    ARENA_target_start               = ARENA_root_start;
    ARENA_target_end                 = ARENA_root_end;

    if(ARENA_root_task_id != -1)
      ARENA_tag[ARENA_TAG_TASK]      = ARENA_root_task_id;
    else
      ARENA_tag[ARENA_TAG_TASK]      = ARENA_NORMAL_TASK;

    ARENA_tag[ARENA_TAG_PARAM]       = ARENA_root_param;
    ARENA_tag[ARENA_TAG_START]       = ARENA_target_start;
    ARENA_tag[ARENA_TAG_END]         = ARENA_target_end;
    ARENA_tag[ARENA_TAG_MORE_FROM]   = -1;
    ARENA_tag[ARENA_TAG_MORE_START]  = -1;
    ARENA_tag[ARENA_TAG_MORE_LENGTH] = -1;
    ARENA_tag_struct recv_tag(ARENA_tag);
    ARENA_recv_list.push(recv_tag);

#ifdef TRACE    
    //traceout(ARENA_local_rank, SEND , TASK , 1, (ARENA_local_rank)%ARENA_nodes, 0);
#endif

  }

  window_buffer = new float[DATA_BUFF_SIZE];

  MPI_Win_create(window_buffer, DATA_BUFF_SIZE*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
  MPI_Win_fence(0, window);

  // Init flags
  ARENA_skip_recv = false;
  ARENA_sent_task = false;

  ARENA_init_data_buff();
}
 
// -----------------------------------------------------------------------
// Initialize data requirement buffer. The buffer is only initialized if
// necessary.
// -----------------------------------------------------------------------
inline void ARENA_init_data_buff() {
  ARENA_data_depend_task = true;
  ARENA_has_data_delivery = true;

  ARENA_local_need_buff  = new float*[ARENA_nodes];

  for(int x=0; x<ARENA_nodes; ++x) {
    ARENA_local_need_buff[x] = new float[DATA_BUFF_SIZE];
  }
}

// -----------------------------------------------------------------------
// Task recv. Will also send out terminator task.
// -----------------------------------------------------------------------
inline int ARENA_task_arrive() {
  //ARENA_sent_task = false;
  ARENA_send_task_tag();
  ARENA_send_task_tag();
  ARENA_send_task_tag();
  ARENA_send_task_tag();

  if(ARENA_recv_list.size() == 0) {// and ARENA_send_list.size() == 0) {
#ifdef DEBUG
    cout<<"[recving] rank "<<ARENA_local_rank<<" is waiting for receiving task tag from "<<(ARENA_local_rank+ARENA_nodes-1)%ARENA_nodes<<endl;
#endif
    MPI_Recv(ARENA_tag, ARENA_TAG_SIZE, MPI_FLOAT, (ARENA_local_rank+ARENA_nodes-1)%ARENA_nodes, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    if(ARENA_tag[ARENA_TAG_TASK] != ARENA_TERMINATE_TASK)
    ARENA_total_task_in += 1;
    ARENA_tag_struct recv_tag(ARENA_tag);
    ARENA_recv_list.push(recv_tag);
#ifdef DEBUG
    cout<<"[enqueued recv list] rank "<<ARENA_local_rank<<" received task "<<ARENA_tag[ARENA_TAG_TASK]<<" start "<<ARENA_tag[ARENA_TAG_START]<<" end "<<ARENA_tag[ARENA_TAG_END]<<" param "<<ARENA_tag[ARENA_TAG_PARAM]<<" terminate_count "<<ARENA_terminate_count<<" recv list size: "<<ARENA_recv_list.size()<<endl;
#endif
  }
  
  ARENA_tag_struct target_tag = ARENA_recv_list.front();
  ARENA_recv_list.pop();
  ARENA_fill_tag(target_tag);
#ifdef DEBUG
  cout<<"[dequeued recv list] rank "<<ARENA_local_rank<<" task "<<ARENA_tag[ARENA_TAG_TASK]<<" start "<<ARENA_tag[ARENA_TAG_START]<<" end "<<ARENA_tag[ARENA_TAG_END]<<" param "<<ARENA_tag[ARENA_TAG_PARAM]<<" terminate_count "<<ARENA_terminate_count<<endl;
#endif

  if(ARENA_tag[ARENA_TAG_TASK] == ARENA_TERMINATE_TASK) {
    if(ARENA_terminate_count > 0 and ARENA_send_list.empty()) {
      ARENA_terminate_count -= 1;
#ifdef DEBUG
      cout<<"[enqueue bypass terminate] rank "<<ARENA_local_rank<<endl;
#endif
      ARENA_tag_struct temp_tag(ARENA_tag);
      ARENA_send_list.push(temp_tag);
    } else if(ARENA_terminate_count <= 0 and ARENA_send_list.empty()) {
      ARENA_total_task_out += 1;
      MPI_Ibsend(ARENA_tag, ARENA_TAG_SIZE, MPI_FLOAT, (ARENA_local_rank+1)%ARENA_nodes, 0, MPI_COMM_WORLD, &request_task);
#ifdef DEBUG
      cout<<"[bypass and terminate] rank "<<ARENA_local_rank<<endl;
#endif
      MPI_Wait(&request_task, &status);
      return ARENA_TERMINATE;
    } else {
      ARENA_terminate_count = ARENA_TERMINATE_TRH;
    }
  } else {
    ARENA_terminate_count = ARENA_TERMINATE_TRH;

#ifdef TRACE
  //traceout(ARENA_local_rank, RECV , TASK , 1, (ARENA_local_rank+ARENA_nodes-1)%ARENA_nodes, ARENA_task_size);
#endif

  }
  return ARENA_CONTINUE;
}

// -----------------------------------------------------------------------
// Task analyze. Analyzing arrival task.
// -----------------------------------------------------------------------
inline void ARENA_task_analyze() {

  ARENA_remote_start       = -1;
  ARENA_remote_end         = -1;
  ARENA_target_id          = -1;
  ARENA_target_start       = -1;
  ARENA_target_end         = -1;
  ARENA_target_more_from   = -1;
  ARENA_target_more_start  = -1;
  ARENA_target_more_length = 0;

  if(ARENA_tag[ARENA_TAG_END] > ARENA_local_end) {
    ARENA_remote_end = ARENA_tag[ARENA_TAG_END];
    ARENA_target_end  = ARENA_local_end;
  } else if(ARENA_tag[ARENA_TAG_END] > ARENA_local_start) {
    ARENA_remote_end = ARENA_local_start;
    ARENA_target_end  = ARENA_tag[ARENA_TAG_END];
  } else {
    ARENA_remote_end = ARENA_tag[ARENA_TAG_END];
  }

  if(ARENA_tag[ARENA_TAG_START] < ARENA_local_start) {
    ARENA_remote_start = ARENA_tag[ARENA_TAG_START];
    ARENA_target_start  = ARENA_local_start;
  } else if(ARENA_tag[ARENA_TAG_START] < ARENA_local_end) {
    ARENA_remote_start = ARENA_local_end;
    ARENA_target_start  = ARENA_tag[ARENA_TAG_START];
  } else {
    ARENA_remote_start = ARENA_tag[ARENA_TAG_START];
  }

  ARENA_target_id = ARENA_tag[ARENA_TAG_TASK];

  ARENA_target_start -= ARENA_local_start;
  ARENA_target_end   -= ARENA_local_start;

  if(ARENA_target_end > ARENA_target_start and ARENA_target_start > -1 and
     ARENA_target_end > -1) {
    ARENA_target_more_from   = ARENA_tag[ARENA_TAG_MORE_FROM];
    ARENA_target_more_start  = ARENA_tag[ARENA_TAG_MORE_START];
    ARENA_target_more_length = ARENA_tag[ARENA_TAG_MORE_LENGTH];
  }
  ARENA_target_param = ARENA_tag[ARENA_TAG_PARAM];
}

// -----------------------------------------------------------------------
// Task dispatch. Send out tasks via bypass and split, but w/o spawn.
// -----------------------------------------------------------------------
inline void ARENA_task_dispatch() {
  if(ARENA_remote_end > ARENA_remote_start) {
    if(ARENA_remote_end <= ARENA_local_start or ARENA_remote_start >= ARENA_local_end) {
      ARENA_tag[ARENA_TAG_START] = ARENA_remote_start;
      ARENA_tag[ARENA_TAG_END]   = ARENA_remote_end;
      
#ifdef TRACE
     //traceout(ARENA_local_rank, SEND , TASK , 1, (ARENA_local_rank+1)%ARENA_nodes, ARENA_task_size);
#endif  

#ifdef DEBUG
      cout<<"[enqueued send list bypass] rank "<<ARENA_local_rank<<" task "<<ARENA_tag[ARENA_TAG_TASK]<<" start "<<ARENA_tag[ARENA_TAG_START]<<" end "<<ARENA_tag[ARENA_TAG_END]<<" param "<<ARENA_tag[ARENA_TAG_PARAM]<<" send list size: "<<ARENA_send_list.size()<<endl;
#endif
      ARENA_tag_struct temp_tag(ARENA_tag);
      ARENA_send_list.push(temp_tag);
    } else {
      ARENA_tag[ARENA_TAG_START] = ARENA_remote_start;
      ARENA_tag[ARENA_TAG_END]   = ARENA_local_start;
#ifdef DEBUG
      cout<<"[enqueued split] rank "<<ARENA_local_rank<<" start "<<ARENA_tag[ARENA_TAG_START]<<" end "<<ARENA_tag[ARENA_TAG_END]<<" param "<<ARENA_tag[ARENA_TAG_PARAM]<<endl;
#endif
      ARENA_tag_struct temp_tag_lower(ARENA_tag);
      ARENA_send_list.push(temp_tag_lower);

      // data synchronization
      ARENA_tag[ARENA_TAG_START] = ARENA_local_end;
      ARENA_tag[ARENA_TAG_END]   = ARENA_remote_end;
#ifdef DEBUG
      cout<<"[enqueued split] rank "<<ARENA_local_rank<<" start "<<ARENA_tag[ARENA_TAG_START]<<" end "<<ARENA_tag[ARENA_TAG_END]<<" param "<<ARENA_tag[ARENA_TAG_PARAM]<<endl;
#endif
      ARENA_tag_struct temp_tag_higher(ARENA_tag);
      ARENA_send_list.push(temp_tag_higher);
    }
    //if(ARENA_send_list.size()>1)
    //  cout<<"normal send_list.size: "<< ARENA_send_list.size()<<endl;

    ARENA_send_task_tag();
    ARENA_send_task_tag();
    ARENA_send_task_tag();
    ARENA_send_task_tag();

#ifdef DEBUG
    cout<<"[sent normal task] rank "<<ARENA_local_rank<<" start "<<ARENA_tag[ARENA_TAG_START]<<" end "<<ARENA_tag[ARENA_TAG_END]<<" param "<<ARENA_tag[ARENA_TAG_PARAM]<<endl;
#endif
  }
}

// -----------------------------------------------------------------------
// Data receive if necessary.
// -----------------------------------------------------------------------
inline void ARENA_data_value_receive() {
  if(ARENA_has_data_delivery and ARENA_target_more_length > 0) {
    // Necessary data receive.

#ifdef DEBUG
    cout<<"[recving] rank "<<ARENA_local_rank<<" is waiting for receiving data from "<<ARENA_target_more_from<<endl;
#endif

    ARENA_total_data_in += ARENA_target_more_length;

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, ARENA_target_more_from, 0, window);
    MPI_Get(ARENA_local_need_buff[ARENA_target_more_from], ARENA_target_more_length, MPI_FLOAT, ARENA_target_more_from, ARENA_target_more_start, ARENA_target_more_length, MPI_FLOAT, window);
    MPI_Win_unlock(ARENA_target_more_from, window);

#ifdef TRACE
    //traceout(ARENA_local_rank, RECV , DATA , 0, ARENA_target_more_from, ARENA_target_more_length*sizeof(float));
#endif      

//    if(ARENA_local_rank == 0) {
//      cout<<"[MPI_Get] rank 0 see what get from "<<ARENA_target_more_from<<" range ("<<ARENA_target_more_start<<" to "<<ARENA_target_more_start + length<<"): ";
//      for (int i=0; i<length; ++i) {
//        cout<<ARENA_local_need_buff[ARENA_target_more_from][i]<<" ";
//      }
//      cout<<endl;
//    }

//    MPI_Recv(ARENA_local_need_buff[ARENA_target_more_from], length, MPI_FLOAT, ARENA_target_more_from, 0, comm_world_data_value, MPI_STATUS_IGNORE);

#ifdef DEBUG
    cout<<"[received] rank "<<ARENA_local_rank<<" received data from "<<ARENA_target_more_from<<" with length "<<ARENA_target_more_length<<endl;
#endif
//    ARENA_store_data(ARENA_target_more_start, ARENA_target_more_length, ARENA_target_more_from, ARENA_local_need_buff[ARENA_target_more_from]);
   ARENA_recv_data_buffer = ARENA_local_need_buff[ARENA_target_more_from];
#ifdef DEBUG
    cout<<"[stored data] rank "<<ARENA_local_rank<<" from "<<ARENA_target_more_from<<" with length "<<ARENA_target_more_length<<endl;
#endif

//    for(int i=0; i<ARENA_nodes; ++i) {
//      if (i != ARENA_local_rank and ARENA_local_need_start[i] != -1) {
//        int length = ARENA_local_need_end[i] - ARENA_local_need_start[i];
//        // we don't use the original data array pointer here since sometimes
//        // the array may contain multiple rows locally, where re-organization
//        // is necessary inside the load/store functions.
//        cout<<"[recving] rank "<<ARENA_local_rank<<" is waiting for receiving data from "<<i<<endl;
//        MPI_Recv(ARENA_local_need_buff[i], length, MPI_FLOAT, i, 0, comm_world_data_value, MPI_STATUS_IGNORE);
//        cout<<"[received] rank "<<ARENA_local_rank<<" received data from "<<i<<" with length "<<length<<endl;
//      }
//    }
//    for(int i=0; i<ARENA_nodes; ++i) {
//      if (i != ARENA_local_rank and ARENA_local_need_start[i] != -1) {
//        int length = ARENA_local_need_end[i] - ARENA_local_need_start[i];
//        ARENA_store_data(ARENA_local_need_start[i], ARENA_local_need_end[i], i, ARENA_local_need_buff[i]);
//        cout<<"[stored data] rank "<<ARENA_local_rank<<" from "<<i<<" with length "<<length<<endl;
//      }
//    }
  }
}

// -----------------------------------------------------------------------
// Task execution.
// -----------------------------------------------------------------------
inline void ARENA_task_exec() {
  if(ARENA_target_end > ARENA_target_start and
     ARENA_target_end > -1 and ARENA_target_start > -1) {
    
    bool require_data = false;
    if(ARENA_target_more_length > 0)
      require_data = true;
    (*(ARENA_kernel_map[ARENA_target_id]))(ARENA_target_start, ARENA_target_end, ARENA_target_param, require_data, ARENA_target_more_length);

#ifdef TRACE    
    //traceout(ARENA_local_rank, EXEC , TASK , ARENA_task_latency, (ARENA_local_rank+ARENA_nodes-1)%ARENA_nodes, ARENA_task_size);
#endif
 
//    if(ARENA_target_id == 1) {
//      new_param = ARENA_kernel(ARENA_target_start, ARENA_target_end, ARENA_target_param);
////    cout<<"[TEST] rank "<<ARENA_local_rank<<" start: "<<ARENA_target_start<<" end: "<<ARENA_target_end<<endl;
//    } else {//if(ARENA_target_id == 2) {
//      new_param = ARENA_kernel1(ARENA_target_start, ARENA_target_end, ARENA_target_param);
//    }
    MPI_Test(&request_profile, &flag_profile, &status_profile);
  }
}

// -----------------------------------------------------------------------
// Data value send.
// -----------------------------------------------------------------------
int ARENA_local_window_pos = 0;
inline int ARENA_data_value_prepare(float* t_start, int t_length) {
//  ARENA_load_data(t_start, t_length, window_buffer);
  for(int i=0; i<t_length; ++i) {
    window_buffer[ARENA_local_window_pos] = *(t_start+i);
    ARENA_local_window_pos += 1;
  }
  ARENA_total_data_out += t_length;
  //MPI_Send(ARENA_remote_ask_buff[i], length, MPI_FLOAT, i, 0, comm_world_data_value);//, &request_data_value);
  //MPI_Ibsend(ARENA_remote_ask_buff[i], length, MPI_FLOAT, i, 0, comm_world_data_value, &request_data_value);
  //MPI_Wait(&request_task, &status);
#ifdef TRACE
//    traceout(ARENA_local_rank, SPAW , DATA , 0, i, t_length*sizeof(float));
#endif  


#ifdef DEBUG
  cout<<"[prepare data] rank "<<ARENA_local_rank<<" from "<<ARENA_local_rank<<" with length "<<t_length<<endl;
#endif
  return ARENA_local_window_pos - t_length;
}

//inline void ARENA_data_value_prepare() {
//
//  if(ARENA_has_data_delivery and ARENA_target_end > ARENA_target_start and
//     ARENA_target_end > -1   and ARENA_target_start > -1) {
//    // Necessary data send
//    for(int i=0; i<ARENA_nodes; ++i) {
//      if (i != ARENA_local_rank and not ARENA_remote_ask_start[i].empty()) {// and ARENA_target_more_end != ARENA_target_more_start) {
//        int temp_start = ARENA_remote_ask_start[i].front();
//        int temp_end = ARENA_remote_ask_end[i].front();
//        ARENA_remote_ask_start[i].pop();
//        ARENA_remote_ask_end[i].pop();
//        int length = temp_end - temp_start;
//        ARENA_load_data(temp_start, temp_end, ARENA_remote_ask_buff[i]);
//        ARENA_total_data_out += length;
//        //MPI_Send(ARENA_remote_ask_buff[i], length, MPI_FLOAT, i, 0, comm_world_data_value);//, &request_data_value);
//        MPI_Ibsend(ARENA_remote_ask_buff[i], length, MPI_FLOAT, i, 0, comm_world_data_value, &request_data_value);
//        //MPI_Wait(&request_task, &status);
//#ifdef DEBUG
//        cout<<"[isent data] rank "<<ARENA_local_rank<<" to "<<i<<" with length "<<length<<endl;
//#endif
//      }
//    }
//  }
//}

// -----------------------------------------------------------------------
// Task issue. Issue the spawned tasks that is customized by users.
// -----------------------------------------------------------------------
inline void ARENA_task_issue() {
  for(int i=0; i<ARENA_num_spawn; ++i) {
    if(ARENA_spawn[i].id != ARENA_NO_TASK) {
      ARENA_fill_tag(ARENA_spawn[i]);
#ifdef DEBUG
      cout<<"[enqueued spawn "<<spawn_count++<<"] rank "<<ARENA_local_rank<<" start "<<ARENA_tag[ARENA_TAG_START]<<" end "<<ARENA_tag[ARENA_TAG_END]<<" param "<<ARENA_tag[ARENA_TAG_PARAM]<<" task type "<<ARENA_tag[ARENA_TAG_TASK]<<endl;
#endif
      if((ARENA_tag[ARENA_TAG_START] >= ARENA_local_end or ARENA_tag[ARENA_TAG_END] <= ARENA_local_start)) {
        ARENA_tag_struct temp_tag(ARENA_tag);
        ARENA_send_list.push(temp_tag);
#ifdef TRACE
      //traceout(ARENA_local_rank, SPAW , TASK , 1, (ARENA_local_rank+1)%ARENA_nodes, ARENA_task_size);
#endif 

      } else {
        ARENA_tag_struct temp_tag(ARENA_tag);
        ARENA_recv_list.push(temp_tag);
        
#ifdef TRACE
      //traceout(ARENA_local_rank, SPAW , TASK , 1, ARENA_local_rank, ARENA_task_size);
#endif

      }
    }
  }

  // reset ARENA_num_spawn
  ARENA_num_spawn = 0;

//  if(!ARENA_sent_task and !ARENA_send_list.empty()) {
  if(!ARENA_send_list.empty()) {
    //if(ARENA_send_list.size()>1)
    //  cout<<"send_list.size: "<<ARENA_send_list.size()<<endl;
    while(ARENA_send_list.size()>0) {
      ARENA_send_task_tag();
    }
  }
//  if(!ARENA_sent_task and ARENA_recv_list.empty()){
//    ARENA_fill_terminate_tag();
////    ARENA_terminate_count = ARENA_TERMINATE_TRH - 1;
//    ARENA_sent_task = true;
//    MPI_Isend(ARENA_tag, ARENA_TAG_SIZE, MPI_FLOAT, (ARENA_local_rank+1)%ARENA_nodes, 0, MPI_COMM_WORLD, &request_task);
//#ifdef DEBUG
//    cout<<"[spawned and sent terminate] rank "<<ARENA_local_rank<<" terminator id: "<<ARENA_tag[ARENA_TAG_PARAM]<<endl;
//#endif
//  }
//  if(!ARENA_sent_task and !ARENA_recv_list.empty()){
//    ARENA_skip_recv = true;
//  }
}

// -----------------------------------------------------------------------
// Fill ARENA_tag based on parameter tag. Note that the additional data
// requirements are also embedded inside the tag format.
// -----------------------------------------------------------------------
inline void ARENA_fill_tag(ARENA_tag_struct t_tag) {
  ARENA_tag[ARENA_TAG_TASK]        = t_tag.id;
  ARENA_tag[ARENA_TAG_START]       = t_tag.start;
  ARENA_tag[ARENA_TAG_END]         = t_tag.end;
  ARENA_tag[ARENA_TAG_PARAM]       = t_tag.param;
  ARENA_tag[ARENA_TAG_MORE_FROM]   = t_tag.more_from;
  ARENA_tag[ARENA_TAG_MORE_START]  = t_tag.more_start;
  ARENA_tag[ARENA_TAG_MORE_LENGTH] = t_tag.more_length;
}

// -----------------------------------------------------------------------
// Create a terminate task tag.
// -----------------------------------------------------------------------
inline void ARENA_fill_terminate_tag() {
  ARENA_tag[ARENA_TAG_TASK]        = ARENA_TERMINATE_TASK;
  ARENA_tag[ARENA_TAG_START]       = -1;
  ARENA_tag[ARENA_TAG_END]         = -1;
  ARENA_tag[ARENA_TAG_PARAM]       = -1;
  ARENA_tag[ARENA_TAG_MORE_FROM]   = -1;
  ARENA_tag[ARENA_TAG_MORE_START]  = -1;
  ARENA_tag[ARENA_TAG_MORE_LENGTH] = -1;
}

// -----------------------------------------------------------------------
// Send a task tag using Ibsend.
// -----------------------------------------------------------------------
inline void ARENA_send_task_tag() {
  if(ARENA_send_list.size()>0) {
    ARENA_tag_struct temp_tag = ARENA_send_list.front();
    ARENA_send_list.pop();
    ARENA_fill_tag(temp_tag);
    ARENA_sent_task = true;
    ARENA_total_task_out += 1;
    MPI_Ibsend(ARENA_tag, ARENA_TAG_SIZE, MPI_FLOAT, (ARENA_local_rank+1)%ARENA_nodes, 0, MPI_COMM_WORLD, &request_task);
  #ifdef DEBUG
    cout<<"[sent] rank "<<ARENA_local_rank<<" start "<<ARENA_tag[ARENA_TAG_START]<<" end "<<ARENA_tag[ARENA_TAG_END]<<" param "<<ARENA_tag[ARENA_TAG_PARAM]<<endl;
  #endif
    MPI_Wait(&request_task, &status);
  }
}

// -----------------------------------------------------------------------
// Spawn a task.
// -----------------------------------------------------------------------
inline void ARENA_spawn_task(int t_id, long long int t_start, long long int t_end, int t_param,
                             float* t_more_start = 0, int t_more_length = 0) {
    ARENA_spawn[ARENA_num_spawn].id          = t_id;
    ARENA_spawn[ARENA_num_spawn].start       = t_start;
    ARENA_spawn[ARENA_num_spawn].end         = t_end;
    ARENA_spawn[ARENA_num_spawn].param       = t_param;

    int window_pos = 0;
    if(t_more_length > 0) {
      window_pos = ARENA_data_value_prepare(t_more_start, t_more_length);
      ARENA_spawn[ARENA_num_spawn].more_from   = ARENA_local_rank;
      ARENA_spawn[ARENA_num_spawn].more_start  = window_pos;
      ARENA_spawn[ARENA_num_spawn].more_length = t_more_length;
    } else {
      ARENA_spawn[ARENA_num_spawn].more_from   = -1;
      ARENA_spawn[ARENA_num_spawn].more_start  = -1;
      ARENA_spawn[ARENA_num_spawn].more_length = -1;
    }

    ARENA_num_spawn++;
}

//inline void ARENA_spawn_task(int t_id, int t_start, int t_end, int t_param) {
//    ARENA_spawn[ARENA_num_spawn].id          = t_id;
//    ARENA_spawn[ARENA_num_spawn].start       = t_start;
//    ARENA_spawn[ARENA_num_spawn].end         = t_end;
//    ARENA_spawn[ARENA_num_spawn].param       = t_param;
//    ARENA_spawn[ARENA_num_spawn].more_from   = -1;
//    ARENA_spawn[ARENA_num_spawn].more_start  = -1;
//    ARENA_spawn[ARENA_num_spawn].more_length = -1;
//    ARENA_num_spawn++;
//}

