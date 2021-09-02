//================================
//helloworld.cpp
//================================
//ARENA implementation of helloworld
//
//

//#define DEBUG
//#define TRACE
#include "../../lib/ARENA.h"
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <ctime>
#include <sys/time.h>
#include <HiParTI.h>
#define NODES 1
#define BLOCK_SIZE 4
#define SOLVE 100
#define DEPENDENT 101
using namespace std;
	
ptiSparseMatrixHiCOO himtx_dev;
ptiValueVector x_dev, b_dev, depend_dev, x_ref;
int startb_dev, endb_dev, total_nb, m, n, mblock, nblock, start_row_dev, end_row_dev, sk_wide; 

int * block_depend_x;
bool * block_depend_y;

ptiNnzIndex nnz_dev;
ptiIndex sb;
ptiIndex sk;

int nodes;
int size;
int local_rank;
uint64_t local_start;
uint64_t local_end;
int local_hicoo_start;

float *local_tran_x;

void ptiGetLtriangular_coo(ptiSparseMatrix *mtx_L, ptiSparseMatrix *mtx){
	ptiNnzIndex nnz_counter = 0;
	ptiIndexVector rowind_tmp; /// row indices, length nnz
    	ptiIndexVector colind_tmp; /// column indices, length nnz
    	ptiValueVector values_tmp; /// non-zero values, length nnz

	ptiNewIndexVector(&rowind_tmp, mtx->nnz, mtx->nnz);   
    	ptiNewIndexVector(&colind_tmp, mtx->nnz, mtx->nnz);
    	ptiNewValueVector(&values_tmp, mtx->nnz, mtx->nnz);
    
	//ptiDumpIndexVector(&mtx->rowind, stdout);

	for (ptiIndex i =0; i < mtx->nnz; i++)
	{
	//printf("index %ld, rowind %ld, colind %d",i , mtx->rowind.data[i], mtx->colind.data[i]);
		if(mtx->rowind.data[i] > mtx->colind.data[i]){
			rowind_tmp.data[nnz_counter] = mtx->rowind.data[i];
			colind_tmp.data[nnz_counter] = mtx->colind.data[i];
			values_tmp.data[nnz_counter] = mtx->values.data[i];
			nnz_counter++;

		}else if (mtx->rowind.data[i]==mtx->colind.data[i]){
			rowind_tmp.data[nnz_counter] = mtx->rowind.data[i];
			colind_tmp.data[nnz_counter] = mtx->colind.data[i];
			values_tmp.data[nnz_counter] = 1.0;
			nnz_counter++;
		}
	}
//	printf("done, nnz_counter %d\n", nnz_counter);
//	fflush(stdout);
	ptiNewSparseMatrix(mtx_L, mtx->nrows,  mtx->ncols, nnz_counter);
	ptiResizeIndexVector(&rowind_tmp, nnz_counter);
	ptiResizeIndexVector(&colind_tmp, nnz_counter);
	ptiResizeValueVector(&values_tmp, nnz_counter);
	ptiCopyIndexVector(&mtx_L->rowind, &rowind_tmp, 1);
	ptiCopyIndexVector(&mtx_L->colind, &colind_tmp, 1);
	ptiCopyValueVector(&mtx_L->values, &values_tmp, 1);
	
	ptiFreeIndexVector(&rowind_tmp);
	ptiFreeIndexVector(&colind_tmp);
	ptiFreeValueVector(&values_tmp);

}

void ptiDistribute_coo_row(ptiSparseMatrix *mtx_dev, ptiSparseMatrix *mtx, int start, int end){
	int nnz_counter = 0;
	ptiIndexVector rowind_tmp; /// row indices, length nnz
  	ptiIndexVector colind_tmp; /// column indices, length nnz
  	ptiValueVector values_tmp; /// non-zero values, length nnz
	ptiNewIndexVector(&rowind_tmp, mtx->nnz, mtx->nnz);   
 	ptiNewIndexVector(&colind_tmp, mtx->nnz, mtx->nnz);
  	ptiNewValueVector(&values_tmp, mtx->nnz, mtx->nnz);
    

	for (int i =0; i < mtx->nnz; i++)
	{
		if((mtx->rowind.data[i] >= start)&&(mtx->rowind.data[i] < end)){
			rowind_tmp.data[nnz_counter] = mtx->rowind.data[i];
			colind_tmp.data[nnz_counter] = mtx->colind.data[i];
			values_tmp.data[nnz_counter] = mtx->values.data[i];
			nnz_counter++;
	}
 }
	ptiNewSparseMatrix(mtx_dev, mtx->nrows,  mtx->ncols, nnz_counter);
  ptiResizeIndexVector(&rowind_tmp, nnz_counter);
	ptiResizeIndexVector(&colind_tmp, nnz_counter);
	ptiResizeValueVector(&values_tmp, nnz_counter);
	ptiCopyIndexVector(&mtx_dev->rowind, &rowind_tmp, 1);
	ptiCopyIndexVector(&mtx_dev->colind, &colind_tmp, 1);
	ptiCopyValueVector(&mtx_dev->values, &values_tmp, 1);
 	ptiFreeIndexVector(&rowind_tmp);
	ptiFreeIndexVector(&colind_tmp);
	ptiFreeValueVector(&values_tmp);


}

void ptiMatrixReallocationHiCOO(ptiSparseMatrixHiCOO *himtx_dev, ptiNnzIndex nnz_dev, int start_block, int end_block, ptiSparseMatrixHiCOO *himtx){

	ptiIndex nb_dev = end_block-start_block;
	int start_nnz = himtx->bptr.data[start_block];

	ptiNnzIndexVector         bptr_tmp;      /// Block pointers to all nonzeros, nb = bptr.length - 1
    	ptiBlockIndexVector       bindI_tmp;    /// Block indices for rows, length nb
    	ptiBlockIndexVector       bindJ_tmp;    /// Block indices for columns, length nb
    	ptiElementIndexVector     eindI_tmp;    /// Element indices within each block for rows, length nnz
    	ptiElementIndexVector     eindJ_tmp;    /// Element indices within each block for columns, length nnz
    	ptiValueVector            values_tmp;      /// non-zero values, length nnz
	

	ptiNewNnzIndexVector(&bptr_tmp, nb_dev+1, nb_dev+1);
	ptiNewBlockIndexVector(&bindI_tmp, nb_dev, nb_dev);
	ptiNewBlockIndexVector(&bindJ_tmp, nb_dev, nb_dev);
	ptiNewElementIndexVector(&eindI_tmp, nnz_dev, nnz_dev);
	ptiNewElementIndexVector(&eindJ_tmp, nnz_dev, nnz_dev);
	ptiNewValueVector(&values_tmp, nnz_dev, nnz_dev);

	bptr_tmp.data[0] = 0;

	for (int i = 0; i < nb_dev; i++){
		bptr_tmp.data[i+1] = himtx->bptr.data[i+1+start_block]-start_nnz;
		bindI_tmp.data[i] = himtx->bindI.data[i+start_block];
		bindJ_tmp.data[i] = himtx->bindJ.data[i+start_block];
	}

	for (int i = 0; i < nnz_dev; i++){
		eindI_tmp.data[i] = himtx->eindI.data[i+start_nnz];
		eindJ_tmp.data[i] = himtx->eindJ.data[i+start_nnz];
		values_tmp.data[i] = himtx->values.data[i+start_nnz];
	}

	ptiNewSparseMatrixHiCOO(himtx_dev, himtx->nrows, himtx->ncols, nnz_dev, himtx->sb_bits, himtx->sk_bits);


	ptiCopyNnzIndexVector(&himtx_dev->bptr, &bptr_tmp);
	ptiCopyBlockIndexVector(&himtx_dev->bindI, &bindI_tmp);
	ptiCopyBlockIndexVector(&himtx_dev->bindJ, &bindJ_tmp);
	ptiCopyElementIndexVector(&himtx_dev->eindI, &eindI_tmp);
	ptiCopyElementIndexVector(&himtx_dev->eindJ, &eindJ_tmp);
	ptiCopyValueVector(&himtx_dev->values, &values_tmp, nnz_dev);

	ptiIndex sk = (ptiIndex)pow(2, himtx->sk_bits);
    	ptiIndex kernel_ndim = (himtx->nrows + sk - 1)/sk;
    	himtx_dev->kschr = (ptiIndexVector*)malloc(kernel_ndim * sizeof(*(himtx->kschr)));

	for(ptiIndex i = 0; i < kernel_ndim; ++i) {
        	ptiCopyIndexVector(&(himtx_dev->kschr[i]), &(himtx->kschr[i]), himtx->nkiters);
    	}

	ptiCopyNnzIndexVector(&himtx_dev->kptr, &himtx->kptr);
	himtx_dev->nkiters = himtx->nkiters;
}


int data_distribution(char  * filename, int nodes, int size)
{


	ptiSparseMatrix mtx, mtx_L, mtx_dev;
	ptiSparseMatrixHiCOO himtx;
      // load mtx data to the hicoo format

	FILE *fi = NULL;
    	fi = fopen(filename, "r");
	ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    	fclose(fi);
    	ptiRandomValueVector(&(mtx.values));    // to better compare results
//	ptiSparseMatrixStatus(&mtx, stdout);

	if(mtx.nrows != mtx.ncols){
		printf("error and exit: matrix' m != n \n");
		return -1;
	}
	
	// Covert to L matrix
	ptiGetLtriangular_coo(&mtx_L, &mtx);
	//if(local_rank == 0) ptiSparseMatrixStatus(&mtx_L, stdout);
   // ptiFreeSparseMatrix(&mtx);
   
  // Convert to HiCOO 
	ptiNnzIndex max_nnzb = 0;
  ptiElementIndex sb_bits = size;
	ptiElementIndex sk_bits = sb_bits+sk_wide;
	sb = (ptiIndex)pow(2, sb_bits);
  sk = (ptiIndex)pow(2, sk_bits);
	ptiAssert(ptiSparseMatrixToHiCOO(&himtx, &max_nnzb, &mtx_L, sb_bits, sk_bits) == 0);
   	// if (ARENA_local_rank == 0) ptiSparseMatrixStatusHiCOO(&himtx, stdout);  
  
  	// Generate x_ref, x_dev and b_dev, block depend

  	ptiNewValueVector(&x_ref, himtx.ncols, himtx.ncols);   //redundent x_ref
  	ptiRandomValueVector(&x_ref);                         
  	ptiNewValueVector(&b_dev, himtx.nrows, himtx.nrows);   //redundent b_dev
  	ptiNewValueVector(&x_dev, himtx.ncols, himtx.ncols);   //redundent x_dev
  	ptiNewValueVector(&depend_dev, himtx.ncols, himtx.ncols);  //redundent x_dev
 

  	printf("Run ptiSparseMatrixMulVectorHiCOO:\n");
  	ptiSparseMatrixMulVectorHiCOO(&b_dev, &himtx, &x_ref); // compute b
        
	//printf("x_ref = [");
  //	for (int i = 0; i< himtx.ncols;i++){
	//	printf("%f ", x_ref.data[i]);
  //	}
       // printf("]\n");
 
	//ptiFreeSparseMatrixHiCOO(&himtx);
   
  	//distribute L matrix
  	m = mtx_L.nrows;
  	n = mtx_L.ncols;
  
 	int balance_block = ceil(ceil((double)m/sk) /nodes); //dense block size //using sk 
  	int balance_row = sk*balance_block;
    int start_row, end_row;
  	start_row =  local_rank*balance_row;
    start_row_dev = (start_row < (m)? start_row : (m));
    end_row = (local_rank+1)*balance_row;
  	end_row_dev = (end_row < (m)? end_row : (m));
  	printf("start row %d, end row %d", start_row, end_row );
  	ptiDistribute_coo_row(&mtx_dev, &mtx_L, start_row, end_row);
// 	ptiSparseMatrixStatus(&mtx_dev, stdout);
  
  //	ptiFreeSparseMatrix(&mtx_L);
  
  
  	//convert to Hicoo_dev
   if(end_row_dev-start_row_dev>0){
  	ptiAssert(ptiSparseMatrixToHiCOO(&himtx_dev, &max_nnzb, &mtx_dev, sb_bits, sk_bits) == 0);
  	//ptiSparseMatrixStatusHiCOO(&himtx_dev, stdout);
   	//ptiFreeSparseMatrix(&mtx_dev);
  }
  	//get dev task token address: startb_dev, endb_dev // dense address space, index = ki*nblock+kj
  	ptiIndex ki_start = ceil((double)start_row/sk);
  	ptiIndex ki_end = ceil((double)end_row/sk);
  	ptiIndex kj = 0;
  	nblock = ceil((double)n/sk);
  	mblock = ceil((double)m/sk);
  	startb_dev = ki_start*nblock+kj;
  	endb_dev = ki_end*nblock+kj;
    total_nb = nblock * mblock;
  	printf("matrix size = (%d, %d) super block size = (%d, %d)\n", m,n,mblock, nblock);
  	block_depend_x = (int *)malloc((nblock*mblock) * sizeof(int)); //setup block_dependency for each block
	block_depend_y = (bool *)malloc((nblock*mblock) * sizeof(bool));
  	for(int i = 0; i < nblock*mblock; i++){
      		block_depend_x[i] = 0;
			block_depend_y[i] = 0;
  	}	
   
   //local_tran_x = (float *)malloc(sb * sizeof(float));
    local_tran_x = (float *)malloc(sk * sizeof(float));

	ptiFreeSparseMatrix(&mtx);
	ptiFreeSparseMatrix(&mtx_L);
	ptiFreeSparseMatrixHiCOO(&himtx);
	ptiFreeSparseMatrix(&mtx_dev);
}

void init_data( ) 
{
    
}

/*

process sptrsv for each block
start = block id
end = block id + 1 
param = block nnz id



*/


// index = row*nblock+cow
// 
//
//

void Solve_hicoo(int start, int end, int param, bool require_data, int length) {


  int globle_id = start+local_start;
 // block_depend[globle_id]++; 

  ptiIndex bi = floor(globle_id /nblock);   //bi:super block row bj:super block column in this case
  ptiIndex bj = globle_id%nblock;

  //param  x = 1 y = 2
  if(param == 1) block_depend_x[globle_id] ++;
  if(param == 2) block_depend_y[globle_id] = 1;
  //ptiIndex sb_dev = (ptiIndex)pow(2, himtx_dev.sb_bits);
  
   if(require_data) {
	//printf("receive x at block (%d %d) from (%d to %d) =", bi, bj, bj*sb, bj*sb+length);  
    for(int i=0; i<length; ++i) {
      x_dev.data[bj*sk+i] = ARENA_recv_data_buffer[i];
	  //printf(" %f ",x_dev.data[bj*sb+i]);
    }
	//printf("\n");
  }
  
  //printf("rank %d, globle_id %d, bi %d, bj %d, dependency %d\n", ARENA_local_rank, globle_id, bi, bj, block_depend[globle_id]);
  
   //find hicoo block ID. To DO ask jiajia 
	ptiIndex himtx_id = 0;

	int hasvalue = 0;
	for(ptiNnzIndex k = 0; k<himtx_dev.kptr.len - 1; k++) {
		ptiIndex  b = himtx_dev.kptr.data[k];
		if((himtx_dev.bindI.data[b]>=bi*sk/sb)&&(himtx_dev.bindI.data[b]<(bi+1)*sk/sb)&&(himtx_dev.bindJ.data[b]>=bj*sk/sb)&&(himtx_dev.bindJ.data[b]<(bj+1)*sk/sb)) {
   			himtx_id = k;
			hasvalue = 1;
   			break;
			}
	}

     if(hasvalue == 0){ //by pass is no value
		 if(bi>bj){  
   			   ptiIndex spawn_block_id = bi*nblock+bi; //spawn to the T
   			  // printf("no value: spawn to row block from %d to %d.\n", globle_id, spawn_block_id );
   			   ARENA_spawn_task(SOLVE, spawn_block_id, spawn_block_id+1, 1); //cannot transmit the left_sum now, need the node ID or transmit it to local	  
		 }
	 }else{

 
  
        //check dependency. process the block if all dependency solve //root block, //solve the first column //solve the matrix multication blocks //solve the solver blocks
        if(((bi==0)&&(bj==0))||((bi>bj)&&(block_depend_y[globle_id]==1))||((bi==bj)&&(block_depend_x[globle_id]==bj))){

      
    
         //spawn ->. 
       // if(bi > bj){
     	 //  ptiIndex spawn_block_id = globle_id+1;
     	   //printf("spawn to row block from %d to %d.\n", globle_id, spawn_block_id );
     	 //  ARENA_spawn_task(SOLVE, spawn_block_id, spawn_block_id+1, 0); //cannot transmit the left_sum now, need the node ID or transmit it to local
       // }
     //}else{ 
     
		 //iter superblock
		   for(ptiIndex b=himtx_dev.kptr.data[himtx_id]; b<himtx_dev.kptr.data[himtx_id+1]; ++b) {

			 ptiNnzIndex block_nnz_id = 0;
			 ptiNnzIndex start_nnz = block_nnz_id+ himtx_dev.bptr.data[b];
     
			 ptiNnzIndex end_nnz = himtx_dev.bptr.data[b+1];
   
			// printf("rank %d , process block %d in super block %d, block_nnz_id %d, start_nnz %d, end_nnz %d \n", ARENA_local_rank, b, himtx_id, block_nnz_id, start_nnz, end_nnz );
    
			 for(ptiNnzIndex j=start_nnz; j < end_nnz; j++)
			 {
   				//printf("start process nnz %d\n", j);
   
				ptiIndex row = himtx_dev.eindI.data[j]+ himtx_dev.bindI.data[b]*sb;
   				ptiIndex col = himtx_dev.eindJ.data[j]+ himtx_dev.bindJ.data[b]*sb;
   
   				//printf("rank %d, start process nnz %d, row %d, col %d, eindI %d, eindJ %d\n", local_rank, j, row, col, himtx_dev.eindI.data[j], himtx_dev.eindJ.data[j]);
       
			   if(row == col){  //solve x
				 //  printf("solve row %d, b %f, left %f, A %f \n", row, b_dev.data[row], x_dev.data[row], himtx_dev.values.data[j] );
   					block_nnz_id++;
   			
   					x_dev.data[row] = (b_dev.data[row]-x_dev.data[row])/himtx_dev.values.data[j];
   				 //	depend_dev.data[row] = 1;
				

   				
           
			  
			   }else if(row>col){ // compute leftsum
   
   					//if(depend_dev.data[col] == 1)
   					//{
   					//	printf("process row %d, nnz %d\n", row, j);
   						x_dev.data[row]+=himtx_dev.values.data[j]*x_dev.data[col];
   
   						block_nnz_id++;
   					//}
			  }else if (row<col){ //upper T
   					//printf("error: row %d < col %d\n", row, col);
   					//return -1;
   				}
			 //}else if((bi==0)&&(bi>bj)&&(block_depend[globle_id]==1)){ //solve the first column
			 //}else if((bi!=0)&&(bi>bj)&&(block_depend[globle_id]==2)){ //solve the matrix multication blocks
			 //}else if((bi==bj)&&(block_depend[globle_id]==1)){ //solve the solver blocks
			 }
		 }
      
	 

		  //spawn ->. 
		  if(bi > bj){
   			   ptiIndex spawn_block_id = bi*nblock+bi;
   			   //printf("spawn to row block from %d to %d.\n", globle_id, spawn_block_id );
   			   ARENA_spawn_task(SOLVE, spawn_block_id, spawn_block_id+1, 1); //cannot transmit the left_sum now, need the node ID or transmit it to local
		  }
      
		  //spawn down
		  if ((bi == bj)&&(bi!=(mblock-1))){
				uint64_t range = local_end - local_start;
				for(int i=local_rank+1; i<nodes; i++){
				//if(local_rank != nodes-1){	
   					float *addr = (x_dev.data + bj*sk);
					ARENA_spawn_task(DEPENDENT, local_end, local_end+1, bj*sk, addr, sk);
					ARENA_spawn_task(DEPENDENT, range*i, range*i+1, bj*sk, addr, sk);
				}
			  //  printf("spawn all colum at %d:", bi);
   			   for(ptiIndex i=bi+1; i < mblock ; i++) {
   				   ptiIndex spawn_block_id = i*nblock+bj;
   				//   printf(" %d", spawn_block_id);
				  float *addr = (x_dev.data + bj*sk);
   				  //ARENA_spawn_task(SOLVE, spawn_block_id, spawn_block_id+1, 2, addr,sk);
				   ARENA_spawn_task(SOLVE, spawn_block_id, spawn_block_id+1, 2 );
   			   }
   			  // printf("\n");
		  }
	  }
  }
  //return -1;
}

/*
int broadcaset_leftSum_hicoo(int start, int end, int param) {

   ptiIndex row = param;
   printf("rank %d update leftsum %d\n", ARENA_local_rank, row);
 
  
return -1;
}

*/

void Dependent_hicoo(int start, int end, int param, bool require_data, int length) {

	 int row = param;
	// int globle_id = start+local_start;
 

	 // ptiIndex bi = floor(globle_id /nblock); 
	 // ptiIndex bj = globle_id%nblock;

	   if(require_data) {
		//printf("receive x at row %d", row);  
		for(int i=0; i<length; ++i) {
		  x_dev.data[row+i] = ARENA_recv_data_buffer[i];
		 // printf(" %f ",x_dev.data[row+i]);
		}
		//printf("\n");
	  }

   //spawn to next node
      //printf("rank %d update dependent %d\n", local_rank, row);
    // if(local_rank != nodes-1){
	//  float *addr = (x_dev.data +  row);
    //  ARENA_spawn_task(DEPENDENT, local_end, local_end+1,  row, addr, sk);
	// }
  
}

       
    
// ----------------------------------------------------------------------
// // Main function. No need to change.
// // ----------------------------------------------------------------------
int main(int argc, char *argv[]) {

       srand(time(0));
       
      nodes =1;
      size =4;
	  sk_wide = 0;
    
    	int argi = 1;
    
    	char  *filename;
    	if(argc > argi)
    	{
        	filename = argv[argi];
        	argi++;
    	}
    
      if((argc > argi)&&(strcmp(argv[argi], "-n") == 0)){
          argi++;
          nodes = stoi(argv[argi]);
          argi++;
      }
      
       if((argc > argi)&&(strcmp(argv[argi], "-s") == 0)){
          argi++;
          size = stoi(argv[argi]);
          argi++;
		  sk_wide = stoi(argv[argi]);
		  argi++;
      }
      
    	// Initialize global data start and end
    	local_rank = ARENA_init(nodes);
  
    	//ex: ./spmv webbase-1M.mtx
     
   	data_distribution(filename,nodes,size);
  
  
  	local_start = startb_dev;
  	local_end   = endb_dev;
  	printf("rank %d, initial addr %lld -- %lld\n", local_rank, local_start, local_end);
  	ARENA_set_local(local_start, local_end);

  	// Register kernel
  	bool isRoot = true;
  	int root_start = 0;
  	int root_end   = 1;
  	int root_param = 0;
  	ARENA_register_task(SOLVE, &Solve_hicoo, isRoot, root_start, root_end, root_param);

	// Register kernel
  	ARENA_register_task(DEPENDENT, &Dependent_hicoo, false); 

	// Initialize local allocated data
    init_data();

   // Execute kernel
    ARENA_run();


   if((end_row_dev == m)&&(start_row_dev != m)){
     int er=0;
     for (int i = 0; i<m; i++){
        if(x_dev.data[i]!=x_ref.data[i])  er++;    
      }
    if(er==0) printf("result pass at rank %d\n", local_rank);
	else {
		printf("x final = [");
		for (int i = 0; i<m; i++){
			printf("%f ", x_dev.data[i]);
		}  
		printf("]\n");

		printf("x_ref = [");
		for (int i = 0; i< m;i++){
			printf("%f ", x_ref.data[i]);
		}
        printf("]\n");
	}
    
 } 
  //	printf("x final = [");
	//for (int i = 0; i<m; i++){
	//	printf("%f ", x_dev.data[i]);
	//}  
	//printf("]\n");

   ptiFreeSparseMatrixHiCOO(&himtx_dev);
  	return 0;
}


