# Grid Project

Grid is a high-performance lattice QCD library developed by Peter Boyle.
Local fork at: /Users/patrickoare/libraries/Grid
Upstream: https://github.com/paboyle/Grid

## User context
The user is a physicist specialising in iterative linear solvers for lattice QCD.
Work is primarily done on the `claude` branch.

## Branch discipline
- Always work on the `claude` branch
- Never push to `develop` or any other branch without explicit user approval

## File editing rules
- NEVER edit or create files without the user's explicit permission
- Always show proposed file contents or diffs and wait for approval before writing

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

## Communication
- Always preface speculative statements (where sources are lacking) with qualifiers like "I think" or "I believe"

## Key areas of interest
- Iterative linear solvers (Krylov subspace methods, CG, GMRES, etc.)
- Grid/`Grid/algorithms/iterative/` and related solver infrastructure
