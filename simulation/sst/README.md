This file introduce how to build the sst-mac and run it with skeleton and dumpi-trace

1. Build SST-mac

1.1 Downloading

SST-macro is available at https://github.com/sstsimulator/sst-macro. You can download the git repository directly:
   git clone https://github.com/sstsimulator/sst-macro.git

1.2 Dependencies

The most critical change is that C++11 is now a strict prerequisite. Workarounds had previously existed for older compilers. These are no longer supported. The following are dependencies for SST-macro.

  Autoconf: 2.68 or later
  
  Automake: 1.11.1 or later
  
  Libtool: 2.4 or later
  
  A C/C++ compiler is required with C++11 support. gcc >=4.8.5 and clang >= 3.7 are known to work.
  
1.3 Build standalone sst/marco

go to the top folder and run:
  ./bootstrap.sh

sets up the configure script:
  ./configure --prefix=$(PATH_TO_INSTALL) CC=mpicc CXX=mpicxx

Build:
  make install

Add $PATH_TO_INSTALL/bin to Path

1.4 build dumpi

go to the dumpi folder in the top folder of sst-mac (not the install path). Tap following to install:
  ./configure --prefix=$(DUMPI_HOME) --enable-libdumpi  
  make 
  
  make install

Add $DUMPI_HOME/lib to LD_LIBRARY_PATH

2 Run 

2.1 configure sst/macro

sst-macro need a configuration file (parameter.ini) to run. I add example files here. More detail please refer to the sst-mac

2.2 Run with skeleton

Go to you application, compile it with sst++ instead of mpic++ to generate the skeleton
run: sstmac -f parameter.ini --exe=(your skeleton)

2.3 Run with dumpi

Go to you application, compile it with -L$(DUMPI_HOME)/lib -ldumpi flag. And run it as normal. 
Then, dumpi will print the MPI trace there which includes: .bin file for the trace and .meta file for the application configureation.

run: dumpi2ascii dumpi-xxxx.bin to see the trace

update the mate file name in `dumpi_parameters.ini` for replaying your trace in sst-mac. Note the `dumpi_parameters.ini` will include the `parameter.ini` file for configuration. please check the include name and path and then run: `sstmac -f dumpi_parameters.ini` to replay

3. Using absolute value for processing time.

replace the original /sst-macro/sstmac/skeletons/undumpi/parsedumpi_callbacks.cc with it in here. setup the timescale as x ns.
 




