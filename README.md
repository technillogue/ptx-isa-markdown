# PTX ISA Documentation + Claude Code Skill

NVIDIA's PTX ISA 9.1 documentation converted to searchable markdown, with a Claude Code skill for GPU development.

## What's Here

1. **PTX ISA 9.1 Documentation** (405 markdown files, 2.3MB)
   - Complete instruction set reference
   - All tables, code blocks, and mathematical notation preserved
   - Organized by chapter with section numbers
   - Images linked to NVIDIA's CDN

2. **CUDA Development Skill** for Claude Code
   - PTX instruction lookup and examples
   - Profiling workflows (nsys, ncu)
   - Debugging patterns (compute-sanitizer, cuda-gdb)
   - TensorCore operation reference (WMMA, WGMMA, TMA)

## Why

The official PTX documentation is a single HTML page. Finding specific instructions or understanding register layouts requires either scrolling through megabytes of HTML or using the browser's search, which doesn't understand context.

This conversion enables:
- `grep -r "register fragment" ptx-docs/` instead of Ctrl+F
- Direct file access for AI tools (Claude, Copilot)
- Offline reference with proper section hierarchy

Example: Find how to disable TMA swizzling:

```bash
$ grep -r "swizzle_mode.*no swizzling" cuda_skill/references/ptx-docs/
9.7.9.28-data-movement-and-conversion-instructionstensormapreplace.md:
  0 | `.u8` | No interleave | No swizzling | 16B | Zero fill
```

Answer in one grep: use `tensormap.replace` with `.swizzle_mode = 0`.

## Structure

```
cuda_skill/                              # Portable Claude Code skill
├── SKILL.md                             # Main skill definition
└── references/
    ├── ptx-docs/                        # 405 markdown files
    │   ├── 9-instruction-set/           # 186 instruction files
    │   │   ├── 9.7.15.5-*.md           # WGMMA register layouts
    │   │   └── 9.7.16-*.md             # TensorCore Gen5 (Blackwell)
    │   ├── 5-state-spaces-types-and-variables/
    │   ├── 8-memory-consistency-model/
    │   └── INDEX.md
    ├── ptx-isa.md                       # Search guide and examples
    ├── nsys-guide.md                    # Nsight Systems patterns
    ├── ncu-guide.md                     # Nsight Compute metrics
    └── debugging-tools.md               # compute-sanitizer, cuda-gdb

scrape_ptx_docs.py                       # Regenerate docs (uv script)
```

## Using the Skill

Install:
```bash
cp -r cuda_skill ~/.claude/skills/cuda
```

The skill activates automatically for CUDA work. Ask Claude:
- "What's the register fragment layout for WGMMA m64n16k16?"
- "How do I disable TMA swizzling?"
- "Profile this kernel with nsys"

Claude searches the local PTX docs and provides answers with section references.

## Search Examples

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

## Regenerating

Run `./scrape_ptx_docs.py` to update from NVIDIA's latest docs. Uses uv for dependency management (beautifulsoup4, html2text, requests).

The scraper:
- Parses the single-page HTML documentation
- Splits by section into individual markdown files
- Preserves tables using markdown syntax
- Converts image references to absolute URLs
- Maintains section hierarchy

## Quality

Tables verified against HTML source. Mathematical notation preserved. 1049 images accessible via NVIDIA CDN links. See `docs/QUALITY_REPORT.md` for detailed verification results.

Known limitations:
- Cross-file anchor links don't resolve (use grep instead)
- Images not downloaded locally (fetch from NVIDIA CDN as needed)

## Technical Details

- PTX ISA Version: 9.1
- Files: 405 markdown files
- Size: 2.3 MB
- Source: https://docs.nvidia.com/cuda/parallel-thread-execution/
- License: Documentation © NVIDIA Corporation

The skill uses Claude Code's progressive disclosure: `SKILL.md` is always loaded (~11KB), reference files load on-demand, and PTX docs are searched rather than loaded into context.

## Use Cases

- Low-level CUDA optimization (inline PTX, instruction selection)
- Understanding compiler output (`cuobjdump -ptx`)
- TensorCore programming (WMMA/WGMMA/TMA operations)
- GPU architecture research
- Training AI models on CUDA/PTX

---

Unofficial conversion for convenience. Refer to [NVIDIA's official documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/) for authoritative reference.
