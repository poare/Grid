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
- Written math conventions: the user writes math in LaTeX, but responses should render math for a client that does NOT typeset inline LaTeX. Use Unicode/plain-text for all inline math (e.g. κ_M(λ), σ_min, γ₅, ‖·‖, ≤, ∠, †, x_λ, s_aug); keep display equations in LaTeX `$$...$$` blocks (those render fine).
- For inline math, avoid combining diacritics (no stacked tildes/bars like H̃, κ̃, λ̄) and keep subscripts to single clean characters. Rename objects to avoid them — e.g. use "aug"/"proj" subscripts (κ_aug, H_aug, I_aug) instead of tilde-decorated symbols, and write complex conjugates as λ* rather than λ̄. Keep display-equation symbol names consistent with the inline names.

## Key areas of interest
- Iterative linear solvers (Krylov subspace methods, CG, GMRES, etc.)
- Grid/`Grid/algorithms/iterative/` and related solver infrastructure
