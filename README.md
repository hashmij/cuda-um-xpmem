This repo contains the basic benchmarks that can serve as proof-of-concept to 
test out the functioning of the GPU-CPU based memory sharing via CUDA unified 
memory and XPMEM's cross-attach-memory. The idea was to access and perform 
operations on GPU memory from a remote process.

I wrote these benchmarks to develop my understanding and answer the following 
questions:

1) How to effectively share the GPU memory between multiple host processes?

2) Understand how the Managed-Memory feature in CUDA works with XPMEM based 
Shared Address Spaces.

3) Understand the differences between x86+PCI vs. NVLINK2+POWER9 for managed 
memory.

4) Are there possibilities for designing efficient intra-/inter-node 
communication runtimes by exploiting CUDA Managed Memory and XPMEM feature for 
both NVLINK as well as non-NVLINK based systems?

Contact: Jahanzeb Hashmi
jahanzeb [dot] maqbool [at] gmail [dot] com
