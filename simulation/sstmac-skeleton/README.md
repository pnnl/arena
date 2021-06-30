# sstmac-skeleton
###
Benchmark from sst-macro.

This benchmark suit includes severial opemMP and MPI benchmarks such as HPCG, lulesh, etc. It requires sst-macro or mpi to compile and execution. When using sst-macro to complie, simply type "make" while in the other cases, revise the CC and CXX in Makefile to mpi compiler. The benchmark suit also include the configuration file for sst-macro. You can ignore it if you do not care about sst-macro.

# TODO:
 - Understand the MPI data movement using the benchmarks. Analysis the data access pattern of baseline case.
 - As the baseline of CFA, testing each of them in sst-macro and evaluating the task-centric implementation. 
 - Adding more kernels as the benchmarks and test it in sst-macro.
