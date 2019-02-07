#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/uio.h>
#include <sys/types.h>
#include <assert.h>
#include <xpmem.h>
#include <cuda_runtime.h>

#include "kernel.h"

/* virtual address range */
#if defined(__x86_64__)
#define MAX_ADDRESS ((uintptr_t)0x7ffffffff000ul)
#else
#define MAX_ADDRESS XPMEM_MAXADDR_SIZE
#endif

/* page alignment */
#define PG_SIZE sysconf(_SC_PAGESIZE)

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

enum transfer_mode {
    HOST=0,
    DEVICE=1
}MODE;

int main(int argc, char **argv)
{
    int i, j;
    int src=0, iters = 100;
    int rank, size;
    char *sbuf, *rbuf;
    ssize_t nread, page_size;
    ssize_t bufsize = 1024;
    double start, end, duration, cost_start, cost_end, cost_duration = 0;
    double min, max, avg, avg_cost, bw;
    int MODE_SRC, MODE_DST;

    xpmem_segid_t segid, *segids;
    struct xpmem_addr addr;
    xpmem_apid_t apid;
    char *attach_ptr;

    /* MPI init */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument processing */
    assert (argc > 2);

    if (strcmp(argv[1], "H")==0 || strcmp(argv[1], "h")==0) {
        MODE_SRC = HOST;
    } else if (strcmp(argv[1], "D")==0 || strcmp(argv[1], "d")==0) {
        MODE_SRC = DEVICE;
    }

    if (strcmp(argv[2], "H")==0 || strcmp(argv[2], "h")==0) {
        MODE_DST = HOST;
    } else if (strcmp(argv[2], "D")==0 || strcmp(argv[2], "d")==0) {
        MODE_DST = DEVICE;
    }
    
    if (argc > 3)
        bufsize = atoi(argv[3]) * 1024;
    if (argc > 4) {
        src = atoi(argv[4]);
    }

    if (rank == 0) {
        printf ("Running with MODE=(%s, %s) with bufsize = %lu\n", (MODE_SRC == HOST) ? 
                "HOST" : "DEVICE", (MODE_DST == HOST) ? "HOST" : "DEVICE", bufsize);
    }

    /* check whether the XPMEM driver is working */
    if (xpmem_version() < 0) {
        fprintf (stderr, "XPMEM driver is not working on the node\n");
    }

    
    page_size = sysconf(_SC_PAGESIZE);
    segids = (xpmem_segid_t *) malloc(sizeof(xpmem_segid_t) * size);

    /* my segment id for XPMEM */
    segid = xpmem_make (0, XPMEM_MAXADDR_SIZE, XPMEM_PERMIT_MODE, (void *)0666);

    /* allocate memory based on given MODE */
    if (rank == src) {
        if (MODE_SRC == DEVICE) {
            cudaMallocManaged((void *)&sbuf, sizeof(bufsize), cudaMemAttachGlobal);
            cudaCheckError();
            do_memset_device(sbuf, bufsize, 'd');
        } else {
            sbuf = (char*)malloc(bufsize);
            posix_memalign((void *)&sbuf, page_size, bufsize);
            memset(sbuf, 's', bufsize);
        }
    } else {
        if (MODE_DST == DEVICE) {
            cudaMallocManaged((void *)&rbuf, sizeof(bufsize), cudaMemAttachGlobal);
            cudaCheckError();
            do_memset_device(sbuf, bufsize, 'd');
        } else {
            rbuf = (char*)malloc(bufsize);
            posix_memalign((void *)&rbuf, page_size, bufsize);
            memset(rbuf, 'r', bufsize);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    /* share the segment id and sbuf with remote processes */
    MPI_Allgather(&segid, sizeof(xpmem_segid_t), MPI_BYTE, segids, sizeof(xpmem_segid_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Bcast(&sbuf, sizeof(char*), MPI_BYTE, src, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
   
    /* attach to remote sbuf and copy data into rbuf */ 
    if (rank != src) { 
        apid = xpmem_get (segids[src], XPMEM_RDWR, XPMEM_PERMIT_MODE, (void*)0666);
        addr.apid = apid;
        addr.offset = (unsigned long)sbuf;
        attach_ptr = xpmem_attach(addr, bufsize, NULL);
    
        /* benchmark */
        start = MPI_Wtime();
        for (i=0; i<iters; i++) {
            /* copy */
            memcpy (rbuf, attach_ptr, bufsize);
        }
        end = MPI_Wtime();
        duration = (end - start)*1e6/iters;
        
        /* Remote process can't launch the kernel on the attached pointer or
         * remote rank's pointer that was used for mallocManaged.
         * This pointer is invalid in current process' context 
         */

        //launch_kernel(attach_ptr, bufsize);
        //launch_kernel(sbuf, bufsize);
        //cudaCheckError(); 
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        //do_print_gpu(sbuf, bufsize);
        //cudaCheckError(); 
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&duration, &min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&duration, &max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&duration, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    avg = avg/size;
    bw = (1.0e6 * bufsize) / (avg * 1024 * 1024);

    if(rank == src) {
        //bw = (1.0 * iters * bufsize * size) / (max*1024*1024);
        printf("nprocs: %d, src_rank: %d, buf: %lu KB, Lat: %.4lf us, bw: %.2lf MBps\n",
                size, src, bufsize, avg, bw);
    } 
    

    MPI_Barrier (MPI_COMM_WORLD);

    /* clean up */
    if (rank == src) {
        cudaFree(sbuf); 
    } else {
        /* detach and release */
        if (NULL != attach_ptr) {
            xpmem_detach(attach_ptr);
        }
        if (0 != apid) {
            xpmem_release(apid);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
