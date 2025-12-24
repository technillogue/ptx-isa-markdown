# NVIDIA CUDA Documentation + Claude Code Skill

NVIDIA's PTX ISA 9.1, CUDA Runtime API 13.1, and CUDA Driver API 13.1 documentation converted to searchable markdown, with a Claude Code skill for GPU development.

## What's Here

1. **PTX ISA 9.1 Documentation** (405 markdown files, 2.3MB)
   - Complete instruction set reference
   - All tables, code blocks, and mathematical notation preserved
   - Organized by chapter with section numbers
   - Images linked to NVIDIA's CDN

2. **CUDA Runtime API 13.1 Documentation** (107 markdown files, 0.9MB)
   - Complete function and data structure reference
   - 41 API modules (device, memory, streams, events, graphs, etc.)
   - 66 data structures (cudaDeviceProp, cudaMemcpy3DParms, etc.)
   - All parameters, return values, and detailed descriptions
   - Navigation, duplicate content, redundant URLs, and boilerplate removed (83% size reduction)

3. **CUDA Driver API 13.1 Documentation** (128 markdown files, 0.8MB)
   - Complete low-level driver API reference
   - 50 API modules (context, module loading, virtual memory, graphs, etc.)
   - 80 data structures (CUdevice, CUcontext, launch parameters, etc.)
   - All parameters, return values, and detailed descriptions
   - Navigation, duplicate TOC, cross-references, and redundant URLs removed (76% size reduction)

4. **CUDA Development Skill** for Claude Code
   - PTX instruction lookup and examples
   - CUDA Runtime & Driver API function reference
   - Profiling workflows (nsys, ncu)
   - Debugging patterns (compute-sanitizer, cuda-gdb)
   - TensorCore operation reference (WMMA, WGMMA, TMA)

## Why

NVIDIA's official documentation is:
- **PTX ISA**: A single 5MB HTML page requiring Ctrl+F through megabytes
- **CUDA Runtime API**: 75+ separate HTML pages requiring multiple clicks
- **CUDA Driver API**: 130+ separate HTML pages requiring multiple clicks

This conversion enables:
- `grep -r "register fragment" ptx-docs/` instead of Ctrl+F
- `grep -r "cudaErrorInvalidValue" cuda-runtime-docs/` instead of clicking through modules
- `grep -r "cuCtxCreate" cuda-driver-docs/` for low-level API lookup
- Direct file access for AI tools (Claude, Copilot)
- Offline reference with proper organization

**Example 1**: Find how to disable TMA swizzling:

```bash
$ grep -r "swizzle_mode.*no swizzling" cuda_skill/references/ptx-docs/
9.7.9.28-data-movement-and-conversion-instructionstensormapreplace.md:
  0 | `.u8` | No interleave | No swizzling | 16B | Zero fill
```

Answer: use `tensormap.replace` with `.swizzle_mode = 0`.

**Example 2**: Look up what cudaErrorInvalidValue means:

```bash
$ grep -A 5 "cudaErrorInvalidValue" cuda_skill/references/cuda-runtime-docs/
```

Answer: Instantly find error code documentation with description and related errors.

**Example 3**: Understand context management in Driver API:

```bash
$ grep -A 20 "cuCtxCreate" cuda_skill/references/cuda-driver-docs/modules/group__cuda__ctx.md
```

Answer: Full cuCtxCreate parameters, return values, and context behavior.

## Structure

```
cuda_skill/                              # Portable Claude Code skill (~4.2MB)
├── SKILL.md                             # Main skill definition
└── references/
    ├── ptx-docs/                        # 405 markdown files (2.3MB)
    │   ├── 9-instruction-set/           # 186 instruction files
    │   │   ├── 9.7.15.5-*.md           # WGMMA register layouts
    │   │   └── 9.7.16-*.md             # TensorCore Gen5 (Blackwell)
    │   ├── 5-state-spaces-types-and-variables/
    │   ├── 8-memory-consistency-model/
    │   └── INDEX.md
    ├── cuda-runtime-docs/               # 107 markdown files (0.9MB)
    │   ├── modules/                     # 41 API modules
    │   │   ├── group__cudart__device.md
    │   │   ├── group__cudart__memory.md
    │   │   ├── group__cudart__stream.md
    │   │   └── ...
    │   ├── data-structures/             # 66 structs/unions
    │   │   ├── structcudadeviceprop.md
    │   │   ├── structcudamemcpy3dparms.md
    │   │   └── ...
    │   └── INDEX.md
    ├── cuda-driver-docs/                # 128 markdown files (0.8MB)
    │   ├── modules/                     # 50 API modules
    │   │   ├── group__cuda__ctx.md
    │   │   ├── group__cuda__mem.md
    │   │   ├── group__cuda__stream.md
    │   │   ├── group__cuda__module.md
    │   │   ├── group__cuda__va.md
    │   │   └── ...
    │   ├── data-structures/             # 80 structs
    │   │   ├── structcudevprop__v1.md
    │   │   ├── structcuda__memcpy3d__v2.md
    │   │   └── ...
    │   └── INDEX.md
    ├── ptx-isa.md                       # PTX search guide and examples
    ├── cuda-runtime.md                  # Runtime API search guide
    ├── cuda-driver.md                   # Driver API search guide
    ├── nsys-guide.md                    # Nsight Systems patterns
    ├── ncu-guide.md                     # Nsight Compute metrics
    └── debugging-tools.md               # compute-sanitizer, cuda-gdb

scrape_ptx_docs.py                       # Regenerate PTX docs (uv script)
scrape_cuda_runtime.py                   # Regenerate Runtime docs (uv script)
scrape_cuda_driver.py                    # Regenerate Driver docs (uv script)
```

## Using the Skill

Install:
```bash
cp -r cuda_skill ~/.claude/skills/cuda
```

The skill activates automatically for CUDA work. Ask Claude:
- "What's the register fragment layout for WGMMA m64n16k16?"
- "How do I disable TMA swizzling?"
- "What does cudaErrorInvalidValue mean?"
- "What fields are in cudaDeviceProp?"
- "How do I create a CUDA context with the Driver API?"
- "What's the difference between cuMemAlloc and cudaMalloc?"
- "Profile this kernel with nsys"

Claude searches the local documentation and provides answers with section references.

## Search Examples

### PTX ISA

Find WGMMA register fragments:
```bash
grep -r "register fragment" cuda_skill/references/ptx-docs/9-instruction-set/ | grep -i wgmma
```

Find all swizzling modes:
```bash
find cuda_skill/references/ptx-docs -name "*swizzl*"
```

Search for any instruction:
```bash
grep -r "mbarrier.init" cuda_skill/references/ptx-docs/
```

### CUDA Runtime API

Look up error code:
```bash
grep -A 10 "cudaErrorInvalidValue" cuda_skill/references/cuda-runtime-docs/
```

Find device properties:
```bash
cat cuda_skill/references/cuda-runtime-docs/data-structures/structcudadeviceprop.md
```

### CUDA Driver API

Look up context creation:
```bash
grep -A 20 "cuCtxCreate" cuda_skill/references/cuda-driver-docs/modules/group__cuda__ctx.md
```

Find virtual memory management:
```bash
ls cuda_skill/references/cuda-driver-docs/modules/*va*.md
cat cuda_skill/references/cuda-driver-docs/modules/group__cuda__va.md
```

Understand module loading:
```bash
grep -A 15 "cuModuleLoad" cuda_skill/references/cuda-driver-docs/modules/group__cuda__module.md
```

Search for stream functions:
```bash
grep -r "cudaStreamSynchronize" cuda_skill/references/cuda-runtime-docs/
```

## Regenerating

Run the scrapers to update from NVIDIA's latest docs:

```bash
# Update PTX ISA docs
./scrape_ptx_docs.py

# Update CUDA Runtime API docs (2-step process for backward compatibility)
./scrape_cuda_runtime.py      # 1. Download and convert
python3 cleanup_cuda_docs.py  # 2. Remove redundant content

# Update CUDA Driver API docs (integrated pipeline with caching)
./scrape_cuda_driver.py       # Download, cache, and cleanup automatically

# Fast iteration on cleanup (uses cached raw files):
./scrape_cuda_driver.py --skip-download
```

All scrapers use uv for dependency management (beautifulsoup4, html2text, requests).

**PTX scraper**:
- Parses single-page HTML documentation
- Splits by section into individual markdown files
- Preserves tables using markdown syntax
- Converts image references to absolute URLs
- Maintains section hierarchy

**Runtime API scraper**:
- Crawls 75+ module and data structure pages
- Organizes into modules/ and data-structures/ directories
- Preserves function signatures, parameters, return values
- Maintains cross-references between functions
- Requires separate cleanup step for backward compatibility

**Driver API scraper**:
- Crawls 130+ module and data structure pages
- Caches raw files to `cuda-driver-docs-raw/` for fast iteration
- Automatically runs cleanup to produce final docs
- Use `--skip-download` to re-run cleanup without re-downloading
- Results in 76% size reduction (3.6MB → 0.8MB)

**API cleanup process**:
- Removes duplicate function TOC (detailed docs remain)
- Removes verbose "See also:" cross-references (grep provides same discoverability)
- Removes anchor links, `[inherited]` tags, zero-width spaces
- Removes footer (Privacy Policy, Copyright, NVIDIA logo)
- Removes redundant URLs from markdown links (type/function names preserved)
- Removes generic boilerplate notes (async errors, initialization errors, callback restrictions)
- Cleans up excessive whitespace
- Runtime API: 83% reduction, Driver API: 77% reduction

## Quality

**PTX ISA**: Tables verified against HTML source. Mathematical notation preserved. 1049 images accessible via NVIDIA CDN links. See `docs/QUALITY_REPORT.md` for detailed verification results.

**CUDA Runtime API**: All 107 pages successfully converted. Function signatures, parameters, return values, and descriptions fully preserved. Type and function names preserved for grep-based navigation. Navigation, duplicate content, redundant URLs, generic boilerplate notes, and formatting noise removed for cleaner, more focused documentation (83% size reduction).

**CUDA Driver API**: All 128 pages (130 total minus 2 404 errors) successfully converted. Function signatures, parameters, return values, and descriptions fully preserved. Type and function names preserved for grep-based navigation. Duplicate function TOC, verbose "See also" sections, redundant URLs, navigation, footer, and formatting noise removed (76% size reduction).

Verification:
- All function parameters documented
- All return values documented
- Data structure fields accessible
- Real-world queries tested and working

Known limitations:
- Cross-file anchor links don't resolve (use grep instead)
- Images not downloaded locally (fetch from NVIDIA CDN as needed)

## Technical Details

**PTX ISA**:
- Version: 9.1
- Files: 405 markdown files
- Size: 2.3 MB
- Source: https://docs.nvidia.com/cuda/parallel-thread-execution/

**CUDA Runtime API**:
- Version: 13.1
- Files: 107 markdown files (41 modules + 66 data structures)
- Size: 0.9 MB (83% reduction from cleaning)
- Source: https://docs.nvidia.com/cuda/cuda-runtime-api/

**CUDA Driver API**:
- Version: 13.1
- Files: 128 markdown files (50 modules + 80 data structures)
- Size: 0.8 MB (76% reduction from cleaning)
- Source: https://docs.nvidia.com/cuda/cuda-driver-api/

**License**: Documentation © NVIDIA Corporation

The skill uses Claude Code's progressive disclosure: `SKILL.md` is always loaded (~13KB), reference files load on-demand, and documentation is searched rather than loaded into context.

**Total skill size**: ~4.2MB (2.3MB PTX + 0.9MB Runtime + 0.8MB Driver + guides)

**Note**: After scraping, the raw cache directory `cuda-driver-docs-raw/` (3.7MB) can be removed from the skill if space is a concern. It's only needed for iterating on cleanup logic.

## Use Cases

- **Low-level CUDA optimization** — Inline PTX, instruction selection
- **Compiler output understanding** — `cuobjdump -ptx` analysis
- **TensorCore programming** — WMMA/WGMMA/TMA operations
- **Runtime API reference** — Error codes, function parameters, struct fields
- **Driver API reference** — Context management, module loading, virtual memory
- **Multi-context applications** — Explicit context control with Driver API
- **PTX/CUBIN module loading** — Dynamic kernel loading at runtime
- **Advanced memory features** — Virtual memory, multicast, tensor maps
- **Context and stream debugging** — Understanding CUDA Runtime behavior
- **Memory management** — Choosing between malloc variants
- **GPU architecture research**
- **Training AI models** on CUDA/PTX/Runtime API

---

Unofficial conversion for convenience. Refer to NVIDIA's official documentation for authoritative reference:
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
