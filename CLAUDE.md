# Grid Project

Grid is a high-performance lattice QCD library developed by Peter Boyle.
Local fork at: /Users/patrickoare/libraries/Grid
Upstream: https://github.com/paboyle/Grid

<!-- General working preferences (user profile, file-editing rules, math and
     communication conventions, research interests) now live in
     ~/research_hub/context/ and load in every session via ~/.claude/CLAUDE.md.
     This file keeps only what is specific to the Grid repository. -->

## Branch discipline
- Always work on the `claude` branch
- Never push to `develop` or any other branch without explicit user approval

## Build
- Build directory: /Users/patrickoare/libraries/Grid/build
- Build examples: `make -C examples` (run from the build directory)
- Most active code lives in Grid/examples/

## Code style
- Follow the existing Grid repository code style as closely as possible
- Model new solver implementations on Grid/Grid/algorithms/iterative/KrylovSchur.h
- Comment code thoroughly — explain what algorithms are doing, not just what the code says
- Prefer clarity over brevity in comments, especially for mathematical/algorithmic steps
- Do NOT duplicate existing code — if a function already exists somewhere in Grid, use it rather than rewriting it
