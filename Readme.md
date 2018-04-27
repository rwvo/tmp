## Kernel instrumentation for branch counting

### Background

One of the problems with doing performance projections for future GPU
hardware based on measurements of current hardware is that the number
of cycles per instruction is different for a subset of the
instructions. Therefore, for executions of code on the baseline
architecture, we need to count for each instruction (e.g.,
`s_mov_b32`, `v_fma_f64`), how often the instruction is executed.

The counting needs to be done on a per-wavefront basis, rather than on
a per-thread basis: on a GPU, when only some of the threads in a
wavefront execute a branch in the code, the other threads that don't
execute the code are masked out: they will execute the instructions in
the branch, but their instructions are not retired. But from an
instruction bandwidth perspective, we still "pay" for the instructions
executed by the masked-out threads. So if at least one of the threads
in a wavefront executes a branch, we pay for all 64
threads. Otherwise, if no thread executes the branch, it is typically
jumped over, and we don't have to count the instructions in the branch.

### Some assembly required: basic blocks

Let's have a look at some GPU assembly:
```asm
BB0_0:
        s_mov_b32 s2, s7
        s_mov_b32 s3, 0
        s_load_dwordx2 s[8:9], s[4:5], 0x0
        s_load_dwordx2 s[10:11], s[4:5], 0x8
        s_load_dwordx2 s[0:1], s[4:5], 0x10
        s_lshl_b64 s[4:5], s[2:3], 4
	;
	; many lines skipped...
	;
        s_mov_b32 s4, 0x100000
        s_mov_b32 m0, -1
BB0_1:                                  ; =>This Inner Loop Header: Depth=1
        v_add_u32_e32 v15, vcc, s2, v1
        v_mov_b32_e32 v0, s3
        v_addc_u32_e32 v16, vcc, v10, v0, vcc
	;
	; many lines skipped...
	;
       s_cbranch_scc1 BB0_1
BB0_2:
        v_add_u32_e32 v0, vcc, v4, v2
        v_addc_u32_e32 v1, vcc, v5, v3, vcc
        v_lshlrev_b64 v[0:1], 3, v[0:1]
        v_mov_b32_e32 v2, s1
        v_add_u32_e32 v0, vcc, s0, v0
        v_addc_u32_e32 v1, vcc, v2, v1, vcc
        flat_store_dwordx2 v[0:1], v[11:12]
        s_endpgm
```

Here, we see a number of _basic blocks_ in the assembly, separated by
labels `BB0_0:` ... `BB0_0:`. Branching and looping is done at the end
of a basic block by jumping forward or backward to a label.

To get a count for the individual instructions, we need to get a count
for the basic blocks, and then multiply the count for each instruction
in the basic block by the count for the block.

### Counting using LDS / shared memory

To get a count for the basic blocks, we insert some assembly at the
start of each basic block that increments a counter in the LDS. For
each wavefront, we will have a separate counter for each basic block.

Incrementing a counter is done in three stages:
1. Read the current counter value from LDS into a register
2. Increment the register value by one
3. Write the updated value back to LDS

Since threads in a wavefront execute in lockstep, any race conditions
are benign: if multiple threads are active, they will all read the
same value, all increment it to the same new value, and all write back
the same updated value. No thread will read an already incremented
counter, and increment is again. That is generally not the case when
threads of multiple wavefronts access the same counter; in that case,
the increment may be either done once or multiple times. That is why
for each basic block, we have a counter for each wavefront.

This gives us the information we need: if at least one thread in a
wavefront executes a basic block, the counter for that wavefront and
basic block is incremented by one. Otherwise, if no thread in the
wavefront executes the basic block, the counter is not updated.

### Macros and tools for instrumenting HCC and HIP code

#### Preparatory steps

This repository provides C++ code and Python tools to instrument the assembly code for the
GPU kernels to do wavefront-level counting of basic blocks in the assembly. Usage instructions:

1. Clone this repository to your development machine. The examples
below assume the repository is cloned in a subdirectory of your code tree, but this is not required.

2. Build your code, and extract the assembly during the link step by
setting an environment variable `KMDUMPISA=1`, or specify the variable
in the build step, e.g., `KMDUMPISA=1 make`. This will generate a file
`dump-gfx803.isa` (number depends on architecure) that contains all
the kernels and their _preambles_..

3. Make a directory to collect the disassembly for each of the GPU
kernels, and extract them from the assembly dump:
```
mkdir original_kernels
branch_counters/grab-all-kernels dump-gfx803.isa original_kernels
```

4. Decide which kernels to instrument.
  - Instrumenting a kernel involves a number of manual steps, and the
process can be tedious if the number of kernels is large. In that
case, it may be easier to instument only the top-n kernels (ordered by
execution time). Skip to step 5 below if you want to instrument all
the kernels. Otherwise:
 - `rcprof -A -w . --hsanokerneldemangle -o trace.atp <program> <program arguments>`
 - `rcprof -T -a trace.atp`
 - open `trace.HSAKernelSummary.html` in a browser
 - kernels are in order of time consumption
 - decide which ones to instrument

5. Matching source code and kernel disassembly
 - To instrument a kernel, we have to match its name in
   `dump-gfx803.isa` (which is the same mangled name as in
   `trace.HSAKernelSummary.html`) to the GPU kernel in the source
   code. We also need to know which number kernel it is in
   `dump-gfx803.isa`, where the first kernel is numbered 0.
 - Do `fgrep hsa_kernel dump-gfx803.isa`. This will list all the
   kernel names in the ISA dump, prepended by
   `.amdgpu_hsa_kernel`. That solves the kernel numbering question. To
   map a mangled kernel name to its source code: the first part of the
   mangled name has the class name (if applicable) and the function
   name in which the kernel appears. In case of doubt, we can insert
   an inline assembly comment in the source code for a GPU kernel:
 - `asm volatile("; LOOK HERE!");`
 - Then, compile again with `KMDUMPISA=1`, and look for the string
   above in `dump-gfx803.isa` to identify the unmangled name of the
   kernel.
 - At this point, we know which GPU kernel we are going to instrument,
   its number is the ISA dump (counting from 0), and the corresponding
   kernel in the source code (i.e., a lamda in a `parallel_for_each`
   in case of HCC code, or a `__global__` function in case of HIP
   code).

#### Adding instrumentation macros to the source code

Briefly documented by example for HIP code below, for the first kernel, named `kernel_0`:

At the top of the source file containing the kernel or the kernel invocation, insert the following.

```
#define HIP_INSTRUMENTATION
#define DO_INSTRUMENTATION
#include "branch_counters/branch_counters.h"
```

Next, we need to figure out somehow what the number of threads in a
workgroup (block, in CUDA speak) is, and then define a compile-time
constant that denoter the number of wavefronts in a workgroup / block:

```
constexpr size_t waves_per_workgroup = TILE * TILE / 64;
```

If in your code, the block size varies from call to call, take the maximum.

Now, instrument the kernel code itself by adding a macro argument to
the kernel, and some macros at the beginning and the end of the kernel:

```
__global__ void matmul_hip_tiled(const NUMBER_T* A, const NUMBER_T* B, NUMBER_T* C
                                 BRANCH_COUNTER_ARG(kernel_0))
{
  BRANCH_COUNTER_GOTO(BB00_0);
  BRANCH_COUNTER_INIT_D(kernel_0, 3, waves_per_workgroup, 2);
  // BRANCH_COUNTER_INC(kernel_0, 0);
  __shared__ NUMBER_T As[tile_size][tile_size];
  __shared__ NUMBER_T Bs[tile_size][tile_size];
  size_t a_x = hipThreadIdx_x;
  size_t a_y = hipBlockIdx_y * tile_size + hipThreadIdx_y;

  // skipping many lines

  C[c_y * MATRIX_SIZE + c_x] = dot_product;
  
  BRANCH_COUNTER_EXIT_D(kernel_0);
  // #include "kernel_0.instr.isa"
  BRANCH_COUNTER_LABEL(BB00_0);
}
```

#### Extracting and instrumenting the assembly for the instrumented source code
1. Rebuild the code with `KMDUMPISA=1`. Don't run this version; it will fail.
2. Extract the assembly for the instrumented kernel (kernel 0 in the example):
```
./branch_counters/grab-kernel.py dump-gfx803.isa 0 > kernel_0.tmp.isa
```
   where the 2nd argument to the script is the number of the kernel in
   the dump, with the first one being 0.
3. Instrument the assembly:
```
./branch_counters/instrument-kernel.py kernel_0.tmp.isa 3 > kernel_0.instr.isa
```
   where the 2nd argument to the script is the number of counters we
   need for this kernel. This number should be identical to the number used in
   the source code.

#### Finalizing the source code instrumentation
In the source code, to back to the bottom of the kernel code, include the instrumented
assembly file, and remove the label. In other words, change this:
```
  BRANCH_COUNTER_EXIT_D(kernel_0);
  // #include "kernel_0.instr.isa"
  BRANCH_COUNTER_LABEL(BB00_0);
```
into this:
```
  BRANCH_COUNTER_EXIT_D(kernel_0);
  #include "kernel_0.instr.isa"
  // BRANCH_COUNTER_LABEL(BB00_0);
```
