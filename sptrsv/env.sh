module load gcc/7.1.0
module load openmpi/4.1.0
module load mkl/2018
#module load cmake/3.15.3
export HIPARTI_PATH=/qfs/people/xiec066/HiParTI/
#export MPIHOME=/qfs/people/xiec066/OpenMPI-3.1.5
#export PATH=$MPIHOME/bin:$PATH
export MPICC=mpicc
export MPICXX=mpicxx
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$HIPARTI_PATH/build:$LD_LIBRARY_PATH
