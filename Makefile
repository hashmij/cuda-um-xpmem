# Paths
P_DIR=$(PWD)
SRC_DIR=$(PWD)/src
OBJ_DIR=$(PWD)/obj
EXE_DIR=$(PWD)/bin

# Dependecy libraries
MPI_INSTALL=/home/hashmij/gpu-work/mvapich2x/install
XPMEM_INSTALL=/opt/xpmem
CUDA_INSTALL=/opt/cuda/9.2

# Compiler
NVCC = $(CUDA_INSTALL)/bin/nvcc
MPICC = $(MPI_INSTALL)/bin/mpicc

CFLAGS = -g -I$(XPMEM_INSTALL)/include -I$(MPI_INSTALL)/include -I$(CUDA_INSTALL)/include
LDFLAGS = -L$(CUDA_INSTALL)/lib64
#LIBS = $(MPI_INSTALL)/lib/libmpi.so $(XPMEM_INSTALL)/lib/libxpmem.so $(CUDA_INSTALL)/lib64/libcudart.so

OBJS = um_kernel.o xpmem_get_um.o

all: $(OBJS)
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OBJS) -lcudart -o xpmem_get_um.x

um_kernel.o: $(SRC_DIR)/um_kernel.cu
	$(NVCC) -c $(SRC_DIR)/um_kernel.cu
	
#xpmem_get_um.o: $(SRC_DIR)/xpmem_get_um.c util.h
xpmem_get_um.o: $(SRC_DIR)/xpmem_get_um.c 
	$(MPICC) -c $(SRC_DIR)/xpmem_get_um.c $(LDFLAGS) -lcudart

# Move executables to ./bin and object files to ./obj
#	mv *.o $(OBJ_DIR)/
#	mv *.x $(EXE_DIR)/

clean:
	rm -f *.o xpmem_get_um.x
#	rm $(EXE_DIR)/*
#	rm $(OBJ_DIR)/*
