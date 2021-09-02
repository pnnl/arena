node=$1
filename=$2
size=$3
sksize=$4

mpirun -n $node  sptrsv_hicoo $filename -n $node -s $size $sksize
