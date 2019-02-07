#/bin/sh


MPI=/home/hashmij/gpu-work/mvapich2x/install
XPMEM_HOME=/opt/xpmem
CUDA_HOME=/opt/cuda/9.2

export PATH=$PATH:$MPI/bin:$XPMEM_HOME/bin:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI/lib:$XPMEM_HOME/lib:$CUDA_HOME/lib64
