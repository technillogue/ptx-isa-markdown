---
name: cuda
description: "CUDA kernel development, debugging, and performance optimization for Claude Code. Use when writing, debugging, or optimizing CUDA code, GPU kernels, or parallel algorithms. Covers non-interactive profiling with nsys/ncu, debugging with cuda-gdb/compute-sanitizer, binary inspection with cuobjdump, and performance analysis workflows. Triggers on CUDA, GPU programming, kernel optimization, nsys, ncu, cuda-gdb, compute-sanitizer, PTX, GPU profiling, parallel performance."
---

# CUDA Programming Skill

## Core Philosophy

**Measure before guessing.** GPU performance is deeply counterintuitive. Profile first, hypothesize second, change third, verify fourth.

**Small, isolated changes.** CUDA bugs compound. Make one change, test it, commit it. Resist the urge to "fix everything at once."

**printf is your strongest tool.** When debuggers fail, when tools produce inscrutable output, printf in device code reveals truth. Don't be embarrassed to use it extensively.

**Sometimes, stare at the diff.** Inscrutable segfaults are common. Tools often don't help. The human approach: minimize the diff, read it carefully, see the bug. This is legitimate and often faster than tooling.

## Debugging Workflow

### First Response to a Bug

1. **Reproduce minimally** — Isolate the failing kernel with smallest possible input
2. **Add printf** — Before any tool, add `printf` in device code to trace execution
3. **Run compute-sanitizer** — Catch memory errors non-interactively:
   ```bash
   compute-sanitizer --tool memcheck ./your_program
   compute-sanitizer --tool racecheck ./your_program  # for race conditions
   compute-sanitizer --tool initcheck ./your_program  # uninitialized memory
   ```
4. **If still stuck**, try cuda-gdb non-interactively for backtrace:
   ```bash
   cuda-gdb -batch -ex "run" -ex "bt" ./your_program
   ```
5. **When tools fail** — Minimize the diff between working and broken code. Read it. The bug is in the diff.

### printf in Device Code

```cuda
__global__ void myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {  // Limit output
        printf("Kernel launched, n=%d, data[0]=%f\n", n, data[0]);
    }
    // ... kernel logic ...
    if (idx < 10) {  // Sample a few threads
        printf("Thread %d: result=%f\n", idx, someValue);
    }
}
```

**Key patterns:**
- Guard with `if (idx == 0)` or `if (idx < N)` to avoid output flood
- Print at kernel entry to confirm launch
- Print intermediate values at suspected failure points
- Flush is automatic at kernel completion

### compute-sanitizer Quick Reference

**Common gotcha:** "Invalid __shared__ write... out of bounds" usually means insufficient dynamic shared memory allocation in the kernel launch, not wrong array indexing. Check `<<<grid, block, smem_size>>>`.

```bash
# Memory errors (most common)
compute-sanitizer --tool memcheck ./program

# Race conditions
compute-sanitizer --tool racecheck ./program

# Uninitialized memory reads
compute-sanitizer --tool initcheck ./program

# Sync errors
compute-sanitizer --tool synccheck ./program

# Save report to file
compute-sanitizer --tool memcheck --log-file report.txt ./program

# Show backtraces (requires -lineinfo compilation)
compute-sanitizer --tool memcheck --show-backtrace yes ./program
```

### cuda-gdb Non-Interactive

```bash
# Get backtrace on crash
cuda-gdb -batch -ex "run" -ex "bt" ./program

# With arguments
cuda-gdb -batch -ex "run arg1 arg2" -ex "bt" ./program

# Examine specific location after crash
cuda-gdb -batch -ex "run" -ex "bt" -ex "info cuda threads" ./program

# Set breakpoint and examine
cuda-gdb -batch \
  -ex "break myKernel" \
  -ex "run" \
  -ex "info cuda threads" \
  -ex "print idx" \
  -ex "continue" \
  ./program
```

**Compile with debug info:**
```bash
nvcc -g -G -lineinfo program.cu -o program
```

### cuobjdump for Binary Inspection

```bash
# Dump PTX
cuobjdump -ptx ./program

# Dump SASS (actual GPU assembly)
cuobjdump -sass ./program

# Show resource usage per kernel
cuobjdump -res-usage ./program

# List all kernels
cuobjdump -symbols ./program | grep -i kernel
```

## Performance Optimization Workflow

### Golden Rule

**Never optimize without profiling first.** Intuition about GPU bottlenecks is almost always wrong.

### Performance Investigation Steps

1. **Establish baseline** — Time the operation, record it
2. **Profile with nsys** — Get timeline, identify which kernels matter
3. **Deep-dive with ncu** — Analyze specific bottleneck kernels
4. **Hypothesize** — Based on metrics, form specific hypothesis
5. **Change one thing** — Make a single targeted change
6. **Verify** — Re-profile, confirm improvement
7. **Repeat**

### nsys (Nsight Systems) — Timeline Profiling

Use nsys for: "Where is time being spent?" — CPU/GPU interaction, kernel launch patterns, memory transfers, overall timeline.

```bash
# Basic profile with report
nsys profile -o report ./program
nsys stats report.nsys-rep

# Focus on CUDA only (faster)
nsys profile --trace=cuda -o report ./program

# Include NVTX markers
nsys profile --trace=cuda,nvtx -o report ./program

# Export to text
nsys stats report.nsys-rep --report cuda_gpu_kern_sum
nsys stats report.nsys-rep --report cuda_api_sum
nsys stats report.nsys-rep --report nvtx_sum

# All available reports
nsys stats report.nsys-rep --help
```

**Key nsys stats reports:**
- `cuda_gpu_kern_sum` — Kernel execution time summary
- `cuda_api_sum` — CUDA API call summary  
- `cuda_gpu_mem_time_sum` — Memory operation times
- `nvtx_sum` — NVTX range summary
- `osrt_sum` — OS runtime (pthread, file I/O, etc.)

### ncu (Nsight Compute) — Kernel Analysis

Use ncu for: "Why is this kernel slow?" — Detailed metrics, roofline, memory analysis, occupancy.

```bash
# Profile all kernels (can be slow)
ncu -o report ./program

# Profile specific kernel
ncu --kernel-name "myKernel" -o report ./program

# Quick summary (no file, prints to stdout)
ncu --set basic ./program

# Full analysis
ncu --set full -o report ./program

# Memory throughput focus
ncu --set memory -o report ./program

# Launch statistics
ncu --set launch -o report ./program

# Roofline analysis
ncu --set roofline -o report ./program

# Show results in CLI
ncu --print-summary per-kernel ./program

# Export to CSV
ncu --csv ./program > metrics.csv
```

**ncu sections (use `--section`):**
- `ComputeWorkloadAnalysis` — Compute throughput
- `MemoryWorkloadAnalysis` — Memory throughput
- `LaunchStatistics` — Blocks, threads, registers
- `Occupancy` — Theoretical vs achieved occupancy
- `SchedulerStatistics` — Warp scheduling
- `SpeedOfLight` — Roofline position
- `SourceCounters` — Per-line metrics (requires -lineinfo)

```bash
# Specific sections
ncu --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis ./program
```

**Warning:** ncu expert system recommendations can be misleading. Always verify with actual metrics and experiments.

### NVTX for Custom Instrumentation

When you need finer granularity than kernel-level, use NVTX:

```cuda
#include <nvtx3/nvToolsExt.h>

void processData() {
    nvtxRangePush("Data Processing");
    
    nvtxRangePush("Preprocessing");
    preprocess();
    nvtxRangePop();
    
    nvtxRangePush("Kernel Execution");
    myKernel<<<grid, block>>>(data, n);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    nvtxRangePush("Postprocessing");
    postprocess();
    nvtxRangePop();
    
    nvtxRangePop();
}
```

**Link with:** `-lnvToolsExt`

Profile and view NVTX stats:
```bash
nsys profile --trace=cuda,nvtx -o report ./program
nsys stats report.nsys-rep --report nvtx_sum
```

### Common Performance Patterns

| Symptom | Likely Cause | Investigation |
|---------|--------------|---------------|
| Low GPU utilization | Kernel launch overhead, CPU bottleneck | nsys timeline, look for gaps |
| Memory bound | Poor access patterns, low cache hit | ncu memory section, check coalescing |
| Compute bound but slow | Low occupancy, register pressure | ncu occupancy, reduce registers |
| Lots of small kernels | Launch overhead dominates | nsys timeline, consider fusion |
| High memcpy time | Excessive H2D/D2H transfers | nsys cuda_gpu_mem, batch transfers |

## Compilation Reference

```bash
# Debug build
nvcc -g -G -lineinfo -O0 program.cu -o program_debug

# Release build
nvcc -O3 -lineinfo program.cu -o program

# Specific architecture
nvcc -arch=sm_80 program.cu -o program  # Ampere
nvcc -arch=sm_89 program.cu -o program  # Ada Lovelace
nvcc -arch=sm_90 program.cu -o program  # Hopper

# Generate PTX (inspect it)
nvcc -ptx program.cu

# Verbose compilation (see register usage)
nvcc --ptxas-options=-v program.cu

# With NVTX
nvcc program.cu -lnvToolsExt -o program
```

**Always compile with `-lineinfo` for production profiling** — minimal overhead, enables source correlation.

## PTX Reference

**Complete PTX ISA 9.1 documentation is available locally** at `references/ptx-docs/` (477 markdown files with all tables, code blocks, and examples preserved).

See `references/ptx-isa.md` for:
- Quick search examples (register fragments, swizzling, TMA)
- Documentation structure guide
- Common PTX patterns
- TensorCore operation references (WMMA, WGMMA, TMA)

Use PTX reference when you need to:
- Understand generated PTX (via `cuobjdump -ptx`)
- Write inline PTX assembly
- Debug code generation issues
- Optimize at the instruction level
- Look up TensorCore operations (WMMA, WGMMA, TMA)
- Understand memory operations and swizzling modes

## Reference Files

- `references/nsys-guide.md` — Detailed nsys usage and analysis patterns
- `references/ncu-guide.md` — Detailed ncu metrics and interpretation  
- `references/debugging-tools.md` — compute-sanitizer, cuda-gdb, cuobjdump details
- `references/nvtx-patterns.md` — NVTX instrumentation patterns
- `references/ptx-isa.md` — PTX instruction set reference (when available)

## Checklist Before Optimizing

- [ ] Established reproducible baseline timing
- [ ] Profiled with nsys to identify hotspots
- [ ] Know which kernel(s) dominate runtime
- [ ] Profiled target kernel with ncu
- [ ] Identified specific bottleneck (memory? compute? latency?)
- [ ] Formed specific, testable hypothesis
- [ ] Plan to change ONE thing
