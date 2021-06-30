<pre>
==========================================
    ___    ____  _______   _____ 
   /   |  / __ \/ ____/ | / /   |
  / /| | / /_/ / __/ /  |/ / /| |
 / ___ |/ _, _/ /___/ /|  / ___ |
/_/  |_/_/ |_/_____/_/ |_/_/  |_|
                                 
==========================================
</pre>

The next generation high-performance computing (HPC) platform is likely to be reconfigurable and data-centric due to the trend of hardware specialization and the emergence of data-driven applications. ARENA is an asynchronous reconfigurable accel- erator ring architecture as a selective solution on how the future HPC cluster will be like. Despite using the coarse-grained reconfigurable arrays (CGRAs) as the substrate platform, our key contribution is not only the static dataflow CGRA design itself, but the ensemble of a new architecture and programming model that enables the asynchronous tasking across a cluster of reconfigurable nodes, so as to bring specialized computation to the data rather than the reverse. We presume distributed data storage, but do not assert any prior knowledge about the exact distribution. Alternatively, hardware specialization for a particular task occurs at runtime when a task verifies the majority of its data are locally available in the present node. In other words, we bring accelerators to their data. The hardware specialization is handled by our high-efficient, fast-configurable CGRAs, while the asynchronous tasking for bring computation to data is achieved by circulating the task token, which describes the dataflow graphs to be executed for a task, among the CGRA cluster connected by a fast ring network.


Related publications
--------------------------------------------------------------------------

- Cheng Tan, Chenhao Xie, Andres Marquez, Antonino Tumeo, Kevin Barker, Ang Li. _"ARENA: Asynchronous Reconfigurable Accelerator Ring to Enable Data-Centric Parallel Computing."_ IEEE Transactions on Parallel and Distributed Systems (TPDS-21).
- Cheng Tan, Chenhao Xie, Ang Li, Kevin Barker, and Antonino Tumeo. _"OpenCGRA: An Open-Source Framework for Modeling, Testing, and Evaluating CGRAs."_ The 38th IEEE International Conference on Computer Design. (ICCD-20), Oct 2020.  [Repo](https://github.com/pnnl/OpenCGRA).


License
--------------------------------------------------------------------------

OpenCGRA is offered under the terms of the Open Source Initiative BSD 3-Clause License. More information about this license can be found here:

  - http://choosealicense.com/licenses/bsd-3-clause
  - http://opensource.org/licenses/BSD-3-Clause



Installation
--------------------------------------------------------

ARENA programming model requires the following additional prerequisites:

 - gcc version gcc/7.1 or later
 - MPI version OpenMPI-3.1.5


Execution
--------------------------------------------------------

This benchmark suite includes BFS, GEMM, SPMV, etc. Just execute `compile.sh` and `run.sh` for compilation and execution, repectively.
Jump into tutorial folder for details.

