#include <stdio.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

#define FIFO 0 
#define LIFO 1


static int coreFreqnecy = 64;     // Fast Memory Latency
static int nwLatency = 64;    // Slow Memory Latency
static int Maxlength = 65536;     // max Queue length

float coreSpeedup = 1;

//Define the structure of memory pages
struct Token {
	int id;
	int local_rank;
	int comment;
	int isTask;
	int arrival_cycle;
	int task_cycle;
	int size;
};


//queue structure
struct taskQueue {
	int rank;
	int length;			     // record dynamic q length
	int stall;
	int totalLength;
	int totalTask;
	int processedTask;
	long int totalLatency;
	int dataBuffer;
	int processinglatency = coreFreqnecy;	//Memory in latency, 
	int Qhead;				// record task at the Qhead head
	int * QtaskID = new int [Maxlength]; // record the task ID
	long int * Qfinish = new long int [Maxlength];	//  record the Arrive time of the task	
	long int * Qissue = new long int [Maxlength];  // record the issue time for the task
	long int * QtaskLatency = new long int [Maxlength];   //  record Q key value 
};

//spaw structure
struct spawBuffer {
	int rank;
	int length;			     // record dynamic q length
	int stall;
	int totalLength;
	int totalTask;
	int processedTask;
	long int totalLatency;
	int * taskID = new int [Maxlength]; // record the task ID
	int * taskDone = new int [Maxlength];
	int * taskTarget = new int [Maxlength];
	int * taskSize = new int [Maxlength];

	//int * Qkey = new int [Maxlength];   //  record Q key value 
};

//network structure
struct ChannelQueue {
	int rank;
	int length;			     // record dynamic q length
	int stall;
	int totalLength;
	int totalIn;
	int totalOut;
	long int totalLatency;
	int transmitlatency = nwLatency; // transmit latency from remote core
	int Qhead;				// record task at the Qhead head
	int * QtaskID = new int [Maxlength]; // record the task ID
	int * Qtail = new int [Maxlength];	//  record the Arrive time of the task	
	int * Qissue = new int [Maxlength];  // record the issue time for the task
	//int * Qkey = new int [Maxlength];   //  record Q key value 
};

//Use to model the memory performance
//Data stored as <IP addr core initialtime\n>
void arenaModeling(string filename, int coreNumber, int policy){


 // File pointer 
    fstream *fin = new fstream [coreNumber];

	 // Open an existing file 
	for (int i=0; i < coreNumber; i++){
		string fileSplit = filename;
		fileSplit.append("."); 
		fileSplit.append(to_string(i));
		//cout<<"openfile " << fileSplit << "\n";
		fin[i].open(fileSplit, ios::in); 
	}
  
    // read the state line
    
    //string line,ip2,addr,core, inittime;

	//initial the channel queue
	//int queueNumber = 2;
	ChannelQueue *cq = new ChannelQueue [coreNumber];   //queue 


	//initial the task queue
	taskQueue *tq = new taskQueue [coreNumber];   //queue 
	
	//initial the spaw queue
	spawBuffer *sb = new spawBuffer [coreNumber]; //buffer

	int * coreStall = new int  [coreNumber];

	for (int i=0; i < coreNumber; i++){
		cq[i].rank = i;
		cq[i].length = 0;
		cq[i].Qhead = 0;
		cq[i].stall = 0;
		cq[i].totalLength=0;
		cq[i].totalIn=0;
		cq[i].totalOut=0;
		cq[i].totalLatency=0;
		for (int j = 0; j < Maxlength; j++){
			cq[i].QtaskID[j] = -1;
			cq[i].Qtail[j] = 0;
			cq[i].Qissue[j] = 0;
			//cq[i].Qkey[j] = 0;
		}

		tq[i].rank = i;
		tq[i].length = 0;
		tq[i].Qhead = 0;
		tq[i].stall = 0;
		tq[i].totalLength=0;
		tq[i].totalTask=0;
		tq[i].processedTask=0;
		tq[i].totalLatency=0;
		tq[i].dataBuffer=0;
		for (int j = 0; j < Maxlength; j++){
			tq[i].QtaskID[j] = -1;
			tq[i].Qfinish[j] = 0;
			tq[i].Qissue[j] = 0;
			tq[i].QtaskLatency[j] = 0;
		}

		sb[i].rank = i;
		sb[i].length = 0;
		sb[i].stall = 0;
		sb[i].totalLength=0;
		sb[i].totalTask=0;
		sb[i].processedTask=0;
		sb[i].totalLatency=0;
		for (int j = 0; j < Maxlength; j++){
			sb[i].taskID[j] = -1;
			sb[i].taskDone[j] = -1;
			sb[i].taskTarget[j] = -1;
			sb[i].taskSize[j] = 0;
		}

		coreStall[i] = 0;

	}

    //int totalinst = 0;

	long int cycle = 0;

	int coreFinish = 0;


	string * line = new string [coreNumber];
	//uint64_t previousAddr =0;
	
	//start fast simulation
    while(coreFinish != coreNumber){    
 
		cycle++; //cycles

		//step 1 process task in task queue

		for(int i = 0; i< coreNumber; i++){
			int targetID = -1;
			bool drain = 0; //if drain this time
				//queue drain on tq[QID]
			if (tq[i].length != 0){	
				//int first = tq[i].Qfinish[0];	
				if (cycle >= tq[i].Qfinish[0]){
					drain = 1;  //task finished
					targetID = tq[i].QtaskID[0];
					//printf("Q %d: time %d Qkey %d Qhead %d\n", q,totalQlatency[q], time, cq[q].Qtail[target], cq[q].Qhead);
				}
				

				//drain the queue
				if (drain == 1){

					//tq[i].totalLatency += cycle - tq[i].Qissue[0];

					tq[i].totalLatency += tq[i].QtaskLatency[0];

					printf("Rank %d exec task done at cycle %d: length %d QtaskID %d Qfinish %d Qissue %d\n", i, cycle, tq[i].length, tq[i].QtaskID[0], tq[i].Qfinish[0], tq[i].Qissue[0]);

					for (int j = 0; j < tq[i].length; j++){   // darin the target
						tq[i].QtaskID[j] = tq[i].QtaskID[j+1];
						tq[i].Qfinish[j] = tq[i].Qfinish[j+1];
						tq[i].Qissue[j] = tq[i].Qissue[j+1];
					}

					//update queue status
					tq[i].Qfinish[0] = cycle + tq[i].QtaskLatency[0];
					tq[i].length--;
					tq[i].processedTask++;
					tq[i].stall=0;
					//update spaw buffer and sent it to output queue
					
					for (int j = 0; j < sb[i].length; j++){
						if(sb[i].taskID[j] == targetID) sb[i].taskDone[j]=1;
					}

				}
					
			}

		}

		//step 2 send task to dispach from spaw buffer
		for(int i = 0; i< coreNumber; i++){
			for (int j = 0; j < sb[i].length; j++){
				if(sb[i].taskDone[j]==1){
					int QID = sb[i].taskTarget[j];
					if((cq[QID].stall != 1)&&(cq[QID].length < Maxlength)){ // sent it to the output queue
						cq[QID].totalIn ++;
						cq[QID].QtaskID[cq[QID].length] = cq[QID].totalIn;
						cq[QID].Qissue[cq[QID].length] = cycle;

						if(QID != i) cq[QID].Qtail[cq[QID].length] = cycle+sb[i].taskSize[j]/cq[QID].transmitlatency;
						else cq[QID].Qtail[cq[QID].length] = cycle;
					//	if (cq[QID].length == 0) cq[QID].Qhead =  cq[QID].Qtail[0]; //if the first in the queue
						
						printf("Rank %d sent spaw to Q %d at cycle %d: length %d QtaskID %d Qtail %d Qissue %d\n", i, QID, cycle, cq[QID].length, cq[QID].QtaskID[cq[QID].length], cq[QID].Qtail[cq[QID].length], cq[QID].Qissue[cq[QID].length]);
						
						cq[QID].length++;

						//pull spaw buffer
						for (int q = j; q < sb[i].length; q++){   // darin the target
							sb[i].taskID[q] = sb[i].taskID[q+1];
							sb[i].taskDone[q] = sb[i].taskDone[q+1];
							sb[i].taskTarget[q] = sb[i].taskTarget[q+1];
						}

						sb[i].length--;
						sb[i].processedTask++;
						sb[i].stall=0;
					}
					else {
						cq[QID].stall=1;
						coreStall[i]=1;
					}  //exit the max length
				}
			}

		}


		//step 3 get new comment from trace	
		for(int i = 0; i< coreNumber; i++){
			//read a comment from trace
			if((!fin[i].eof())&&(coreStall[i]==0)){
				getline(fin[i], line[i]);
				//cout<<"get line from file:  "<< line[i]<<endl;
			}
			std::stringstream s(line[i]);
			string localRank, comment, isTask, taskLatency, remoteRank, size;
			getline(s, localRank,' ');
			getline(s, comment,' ');
			getline(s, isTask,' ');
			getline(s, taskLatency,' ');
			getline(s, remoteRank,' ');
			getline(s, size,' ');
			//Token newToken; 
			
			if((comment=="Recv")&&(isTask=="Task")){   //receive a task
				//queue drain from cq[localRank].QtaskID[0]
				int QID = stoi(localRank);
				if ((cycle > cq[QID].Qtail[0])&&(cq[QID].length != 0)){

					cq[QID].totalLatency += cycle - cq[QID].Qissue[0];

					printf("Rank %d receive task from Q %d at cycle %d: length %d, QtaskID %d Qtail %d Qissue %d\n", QID, QID, cycle, cq[QID].length, cq[QID].QtaskID[0], cq[QID].Qtail[0], cq[QID].Qissue[0]);
					//update queue status
					for (int j = 0; j < cq[QID].length; j++){   // darin the target
						cq[QID].QtaskID[i] = cq[QID].QtaskID[i+1];
						cq[QID].Qtail[i] = cq[QID].Qtail[i+1];
						//cq[QID].Qkey[i] = cq[QID].Qkey[i+1];
						cq[QID].Qissue[i] = cq[QID].Qissue[i+1];
					}	
					cq[QID].length--;
					cq[QID].totalOut ++;
					cq[QID].stall=0;
					coreStall[stoi(localRank)]=0;
				}else{
					coreStall[stoi(localRank)]=1;
				}

			}else if((comment=="Send")&&(isTask=="Task")){  //send a task
				//queue insert on cq[QID]
				int QID = stoi(remoteRank);
				if ((cq[QID].length < Maxlength)&&(cq[QID].stall!=1)){
					cq[QID].totalIn ++;
					cq[QID].QtaskID[cq[QID].length] = cq[QID].totalIn;
					cq[QID].Qissue[cq[QID].length] = cycle;

					if(remoteRank != localRank) cq[QID].Qtail[cq[QID].length] = cycle+stoi(size)/cq[QID].transmitlatency;
					else cq[QID].Qtail[cq[QID].length] = cycle;
				//	if (cq[QID].length == 0) cq[QID].Qhead =  cq[QID].Qtail[0]; //if the first in the queue
						
					printf("Rank %d send task to Q %d at cycle %d: length %d QtaskID %d Qtail %d Qissue %d\n", stoi(localRank), QID, cycle, cq[QID].length, cq[QID].QtaskID[cq[QID].length], cq[QID].Qtail[cq[QID].length], cq[QID].Qissue[cq[QID].length]);
						
					cq[QID].length++;		
				}
				else {
					cq[QID].stall=1;
					coreStall[stoi(localRank)]=1;
				}  //exit the max length

			}else if((comment=="Exec")&&(isTask=="Task")){  // add the task to queue
				//queue insert on cq[QID]
				int QID = stoi(localRank);
				if ((tq[QID].length < Maxlength)&&(tq[QID].stall!=1)){
					tq[QID].totalTask++;
					tq[QID].QtaskID[tq[QID].length] = tq[QID].totalTask;
					tq[QID].Qissue[tq[QID].length] = cycle;
					tq[QID].Qfinish[tq[QID].length] = cycle+(long) (stol(taskLatency)/coreSpeedup) + tq[QID].dataBuffer;
					tq[QID].dataBuffer = 0;
					tq[QID].QtaskLatency[tq[QID].length] = (long) (stol(taskLatency)/coreSpeedup);
						
					printf("Rank %d exec task to queue %d at cycle %d: length %d QtaskID %d Qtail %d Qissue %d\n", stoi(localRank), QID, cycle, tq[QID].length, tq[QID].QtaskID[tq[QID].length], tq[QID].Qfinish[tq[QID].length], tq[QID].Qissue[tq[QID].length]);
						
					tq[QID].length++;		
				}
				else {
					tq[QID].stall=1;
					coreStall[QID]=1;
				}  //exit 
			}else if((comment=="Spaw")&&(isTask=="Task")){  // add the task to spwa queue
				//queue insert on cq[QID]
				int QID = stoi(localRank);
				if (sb[QID].length < Maxlength){
					sb[QID].totalTask ++;
					sb[QID].taskID[sb[QID].length] = tq[QID].totalTask;
					sb[QID].taskDone[sb[QID].length] = 0;
					sb[QID].taskTarget[sb[QID].length] =  stoi(remoteRank);	
					sb[QID].taskSize[sb[QID].length] = stoi(size);
					printf("Rank %d spaw task to spaw buffer %d at cycle %d: length %d QtaskID %d\n", stoi(localRank), QID, cycle, sb[QID].length, sb[QID].taskID[sb[QID].length]);
						
					sb[QID].length++;		
				}
				else {
					sb[QID].stall=1;
					coreStall[stoi(localRank)]=1;
				}

			}else if((comment=="Recv")&&(isTask=="Data")){
				int QID = stoi(localRank);
				tq[QID].dataBuffer += stoi(size)/cq[QID].transmitlatency;
				printf("Rank %d waiting %d data to arrive from Rank %d\n", QID, stoi(size),  stoi(remoteRank));
			}else if((comment=="Send")&&(isTask=="Data")){
			}



			

		}
		
		//step 4: check all comment acesses have been proccessed
		coreFinish = 0;
		for(int i = 0; i< coreNumber; i++){
			//printf("Q %d total inst %d processed %d\n", q ,totalinstQ[q],processedinstQ[q]);
			if((fin[i].eof())&&(tq[i].processedTask == tq[i].totalTask)){
				//printf("Q %d finished %d",q, Qfinish);
				coreFinish++;
			}
		}
	}

	printf("\n========simulation end===========\n");

	long int totaltaskLatency = 0;
	int totalTask = 0;

	for(int i = 0; i<coreNumber; i++){
		
		printf("core %d, total task lantency is %d\n", i, tq[i].totalLatency);
		totaltaskLatency += tq[i].totalLatency;
		totalTask += tq[i].totalTask;

		printf("channel queue %d latency is %f\n", i, (double)cq[i].totalLatency/(double)cq[i].totalIn);
	}

	printf("average task latency is %d\n", totaltaskLatency/totalTask);
	printf("average core latency is %d\n", totaltaskLatency/coreNumber);


	for(int i = 0; i< coreNumber; i++){
	 fin[i].close();
	}
}

//main test 
int main(int argc, char ** argv){



   printf("---------------------------------------------------------------------------------------------\n");
   int argi = 1;
   char *tracefile;
    if(argc > argi)
    {
        tracefile = argv[argi];
        argi++;
		if (strcmp(tracefile, "--help") == 0){
			printf("using ./arena-sim (trace) [-n #coreNumber] [--policy queue policy]\n");
			return -1;
		}
    }
        
	char *qpoint;
	int policy = 0;
	int coreNumber = 0;
	
	//get parameters
    while(argc > argi){
    
        qpoint = argv[argi];
        argi++;
    
        
		if (strcmp(qpoint, "-n") == 0){
		printf("configuration core number = ");
		coreNumber = atoi(argv[argi]);
 		printf("%d\n",  coreNumber);
		argi++;		
		}

		if (strcmp(qpoint, "-s") == 0){
		printf("configuration core speed = ");
		coreSpeedup = atof(argv[argi]);
 		printf("%f\n",  coreSpeedup);
		argi++;		
		}
        
		if (strcmp(qpoint, "--policy") == 0){
		printf("configuration queue policy = ");
		policy = atoi(argv[argi]);
		switch(policy){ 
		case 0:	printf("FIFO\n"); break;
		case 1:	printf("LIFO\n"); break;
		case 2:	printf("PAGEHIT\n"); break;
		default: printf("not configure, used FIFO\n"); break;
		}
		argi++;		
		}

   }

   std::cout << "Follow this application: " << tracefile << "\n";
		 
   arenaModeling( tracefile, coreNumber, policy);
   
   return 0;  
}
