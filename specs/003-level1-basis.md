# 003 – level1-basis

## Goal

For the free scalar theory (λ = 0 of spec 002) on a periodic T × L lattice, provide the
level-1 basis ψ_j = Π(j) = Σ_x z(j,x), j = 0..T-1 (GALERKIN.md §3.3), its gradient and
Laplacian, and the Stein operator ℒψ_j = ∇²ψ_j − ∇S·∇ψ_j (GALERKIN.md §1.4, §3.1).
Each quantity is a pure function of one (T, L) configuration, with a batch version over
(T, L, N). The tests compare values and derivatives with hand values and ForwardDiff;
matrices, solvers and residuals are the next spec.

## Interface

Layout: a single configuration `z` has size `(T, L)` (time slab `t`, site `x`); a batch has
size `(T, L, N)`, configuration index last (same as specs 001 and 002). `L` is the spatial
size (N_s in GALERKIN.md); ℒ denotes the Stein operator. `M` is the number of basis
functions; level 1 requires `M == T`. Index `j = 0..T-1` is Julia index `j+1`.

Each name below is one function with three methods, dispatched on the rank of `z`:
the `(T, L)` method is the pure single-configuration routine; the `(T, L, N)` method is
`stack(map(<(T,L) method>, eachslice(z; dims=3)))`, so the single-configuration result is
stacked on the last dimension. Any other rank throws `ArgumentError`.

```julia
basis(z::AbstractMatrix{<:Real}, M)                 -> Vector, size (M,)
    # ψ_j = sum(selectdim(z, 1, j+1))
basis(z::AbstractArray{<:Real,3}, M)                -> Matrix, size (M, N)

basis_grad(z::AbstractMatrix{<:Real}, M)            -> Array, size (T, L, M)
    # [t+1, x+1, j+1] = ∂ψ_j/∂z(t,x) = δ_tj
basis_grad(z::AbstractArray{<:Real,3}, M)           -> Array, size (T, L, M, N)

basis_laplacian(z::AbstractMatrix{<:Real}, M)       -> Vector, size (M,)
    # ∇²ψ_j = 0 for every j
basis_laplacian(z::AbstractArray{<:Real,3}, M)      -> Matrix, size (M, N)

stein_psi(z::AbstractMatrix{<:Real}, M, p)          -> Vector, size (M,)
    # ℒψ_j = ∇²ψ_j − ∇S·∇ψ_j = −sum(selectdim(grad_slice(p...)(z), 1, j+1))
stein_psi(z::AbstractArray{<:Real,3}, M, p)         -> Matrix, size (M, N)
```

`p = (λ, κ)` is the tuple of arguments of `grad_slice` (spec 002); `stein_psi` reuses
`grad_slice(p...)` and does not copy its formula. The formula for ℒψ_j holds for any λ
(ψ_j is linear), but only λ = 0 is specified and tested; the tests always pass `λ = 0`.

All eight methods are pure: no mutation of `z`, no I/O. Element types follow the inputs
(`z`, and `λ`, `κ` for `stein_psi`); nothing is forced to `Float64`, so that ForwardDiff
dual numbers pass through. `basis_grad` and `basis_laplacian` build their constants with
`eltype(z)`. The slab sum uses `sum(selectdim(z, 1, t))` on the `(T, L)` slice; `slab_sums`
(spec 001) is not used. Names `basis`, `basis_grad`, `basis_laplacian`, `stein_psi` are
exported from `src/GalerkinDemo.jl`.

Input validation (throws `ArgumentError`, never returns NaN silently): rank of `z` not 2
or 3; `T < 1`, `L < 1` or (batch) `N < 1`; `M ≠ T`. No finiteness check.

## Formulas

Indices: time `t ∈ {0..T-1}` (array index `t+1`), site `x ∈ {0..L-1}` (array index `x+1`),
configuration `n ∈ {1..N}`. Both directions are periodic: `t±1` is taken modulo `T`,
`x±1` modulo `L`. `δ_ab` is the Kronecker delta. All formulas below are for one
configuration `z(t,x)`; the batch methods apply them to `z[:,:,n]` for each `n`.

**Action and force.** The action `S` and the force `F(t,x) = ∂S/∂z(t,x)` are those of
spec 002 (`ϕ` there is `z` here; `F` is `grad_slice(λ, κ)(z)`); they are not restated.
This spec uses `λ = 0`.

**Basis** (`basis`). For `j = 0..T-1` (`M = T`)

    ψ_j(z) = Π(j) = Σ_x z(j,x)                (slab sum of time slice j)

**Gradient** (`basis_grad`)

    ∂ψ_j/∂z(t,x) = δ_tj

independent of `z` and of `x`. Hence, with `∇ψ_i·∇ψ_j := Σ_{t,x} (∂ψ_i/∂z(t,x)) (∂ψ_j/∂z(t,x))`,

    ∇ψ_i·∇ψ_j = Σ_{t,x} δ_ti δ_tj = L δ_ij.

**Laplacian** (`basis_laplacian`)

    ∇²ψ_j = Σ_{t,x} ∂²ψ_j/∂z(t,x)² = 0         (ψ_j is linear in z)

**Stein operator** (`stein_psi`). For a scalar function Φ of `z`

    ℒΦ = ∇²Φ − ∇S·∇Φ,        ∇S·∇Φ = Σ_{t,x} F(t,x) ∂Φ/∂z(t,x).

For `Φ = ψ_j`, using the gradient and the Laplacian above,

    ℒψ_j = 0 − Σ_{t,x} F(t,x) δ_tj = − Σ_x F(j,x).                          (★)

(★) holds for any `λ`; `stein_psi(z, M, p)` with `p = (λ, κ)` returns it with
`F = grad_slice(λ, κ)(z)`.

**Closure at `λ = 0`.** There the spec-002 force is `F = 2z − 2κ (sum of the four
nearest neighbours of z)`. Summing row `j` over `x`, each spatial neighbour term
`z(j,x±1)` sums to `Π(j)`, so

    Σ_x F(j,x) = 2Π(j) − 2κ (Π(j−1) + Π(j+1) + 2Π(j)) = (2 − 4κ) Π(j) − 2κ (Π(j−1) + Π(j+1)),

that is

    ℒψ_j = − Σ_i K0_ij ψ_i,
    K0_ij = (2 − 4κ) δ_ij − 2κ (δ_{i,j+1} + δ_{i,j−1})      (i, j = 0..T-1, periodic in time).   (★★)

`K0` is the `T × T` periodic matrix; for `T ≥ 3` it is symmetric circulant with column sums
`2 − 8κ`. In particular for a constant field `z ≡ c` every `ψ_i = L c`, so
`ℒψ_j = −L (2 − 8κ) c` for every `j`.

## Acceptance criteria

Common setup (unless an AC says otherwise): `T = 4`, `L = 3`, `M = T`, `κ = 0.1` (`< 1/4`),
`λ = 0`, so `p = (0.0, 0.1)` in every call of `stein_psi`; `T ≠ L` on purpose, to catch
swapped axes, and `T, L ≥ 3`. Random data: `rng = Random.MersenneTwister(1234)` and
`z = randn(rng, 4, 3)` (single configuration). "Exact" means
`isapprox(a, b; rtol=1e-14, atol=1e-14)`; "to 1e-12" means `rtol=1e-12, atol=1e-12`
(the Laplacian is 0, so a relative tolerance alone would fail). Index `j = 0..T-1` is
Julia index `j+1`. The tests use only the names of the Interface, plus `action_slice` and
`grad_slice` (spec 002), `ForwardDiff` (test dependency, as in 002) and `LinearAlgebra.tr`.
Test helpers live in the test file, not in `src/`. Each AC is one `@testset`
"AC-<n>: <short title>"; every AC also checks `size` of what it compares.

- **AC-1: hand anchors.** Formulas used: `ψ_j = Σ_x z(j,x)` and (GALERKIN.md §3.2–3.3)
  `ℒψ_j = −Σ_x F(j,x)`, which for a constant field `z ≡ c` is `−L (2 − 8κ) c`. Exact.
  (a) `T = 4`, `L = 3`, `z[t+1, x+1] = t + 1`: `ψ_j = 3 (j+1)`, so
  `basis(z, 4) == [3, 6, 9, 12]`.
  (b) `z = fill(2.0, 4, 3)`, `κ = 0.1`, `λ = 0`: `stein_psi(z, 4, (0.0, 0.1))` is
  `−3 · (2 − 0.8) · 2 = −7.2` for every `j`, i.e. `fill(-7.2, 4)`.
- **AC-2: derivatives vs ForwardDiff.** Random `z`. Formulas used (GALERKIN.md §3.3):
  `∂ψ_j/∂z(t,x) = δ_tj`, `∇²ψ_j = Σ_{t,x} ∂²ψ_j/∂z(t,x)² = 0`,
  `∇ψ_i·∇ψ_j = Σ_{t,x} δ_ti δ_tj = L δ_ij`. With `ψ_j(w) = basis(w, 4)[j+1]`, to 1e-12:
  `basis_grad(z, 4)[:,:,j+1]` equals `ForwardDiff.gradient(ψ_j, z)` for each `j`;
  `basis_laplacian(z, 4)[j+1]` equals `tr(ForwardDiff.hessian(ψ_j, z))` for each `j`
  (both are `0`); with `G = basis_grad(z, 4)`, `sum(G[:,:,i+1] .* G[:,:,j+1])` equals
  `3 δ_ij` for all `i, j`. Sizes: `basis_grad` `(4,3,4)`, `basis_laplacian` `(4,)`.
- **AC-3: Stein operator vs ForwardDiff.** Random `z`, `κ = 0.1`, `λ = 0`. Formula used
  (GALERKIN.md §1.4): `ℒψ_j = tr Hess ψ_j − ∇S·∇ψ_j`, with `∇S·∇ψ_j = Σ_{t,x}
  (∂S/∂z(t,x)) (∂ψ_j/∂z(t,x))`. Here `∇S = ForwardDiff.gradient(action_slice(0.0, 0.1), z)`
  (gradient of the action, not `grad_slice`), and `∇ψ_j`, `tr Hess ψ_j` come from
  ForwardDiff of `ψ_j(w) = basis(w, 4)[j+1]`. Then `stein_psi(z, 4, (0.0, 0.1))[j+1]`
  equals `tr(ForwardDiff.hessian(ψ_j, z)) − sum(∇S .* ForwardDiff.gradient(ψ_j, z))`
  for each `j`, to 1e-12. Size `(4,)`.
- **AC-4: closure.** Same `z`. Formula used (GALERKIN.md §3.2, §3.3, restated in Formulas
  (★★)): `ℒψ_j = −Σ_i K0_ij ψ_i`, `K0_ij = (2 − 4κ) δ_ij − 2κ (δ_{i,j+1} + δ_{i,j−1})`,
  indices mod `T`. The test builds the `4 × 4` periodic matrix `K0` itself (helper in the
  test file, with `κ = 0.1`). Then `stein_psi(z, 4, (0.0, 0.1))` equals
  `-(K0' * basis(z, 4))` (entry `j`: `−Σ_i K0[i,j] ψ_i`), to 1e-12.
- **AC-5: batch consistency, shapes, independence.** `zb = randn(rng, 4, 3, 5)`
  (`N = 5 ≠ T, L`, a fresh `rng = MersenneTwister(1234)`). Formula used (Interface): the
  batch method is the single-configuration method applied to `zb[:,:,n]` and stacked on the
  last dimension. For `f ∈ {basis, basis_grad, basis_laplacian, stein_psi}`, with
  `p = (0.0, 0.1)` for `stein_psi`: sizes are `(4,5)`, `(4,3,4,5)`, `(4,5)`, `(4,5)`;
  `selectdim(f(zb, 4), ndims(f(zb, 4)), n) == f(zb[:,:,n], 4)` for every `n`, exact
  (`==`). Replacing `zb[:,:,2]` by other random values leaves the results for `n ≠ 2`
  unchanged, exact (`==`).
- **AC-6: invalid input.** Formula used (Interface, Validation): `ArgumentError` for rank
  of `z` not 2 or 3, `T < 1`, `L < 1`, batch `N < 1`, `M ≠ T`. Each of the four functions
  throws `ArgumentError` (with `p = (0.0, 0.1)` for `stein_psi`) for: `z = randn(12)`
  (rank 1); `z = randn(4,3,5,2)` (rank 4); `z` of size `(4,3)` with `M = 3` and `M = 5`;
  `z = zeros(0,3)`; `z = zeros(4,0)`; `z = zeros(4,3,0)`.
- **AC-7: no mutation, element type, dual numbers.** Formulas used: Interface (pure
  functions, element types follow the inputs) and linearity of `ℒψ_j` in `z` at `λ = 0`
  (by (★★), no constant term). (a) `z` and `zb` equal a copy taken before the calls
  (`==`) after all eight methods. (b) With `z32 = Float32.(z)` and `p32 = (0f0, 0.1f0)`,
  the results of `basis`, `basis_grad`, `basis_laplacian` and `stein_psi` all have
  `eltype == Float32`. (c) `ForwardDiff.derivative(c -> sum(stein_psi(c .* z, 4, (0.0, 0.1))), 1.0)`
  equals `sum(stein_psi(z, 4, (0.0, 0.1)))`, to 1e-12 (duals pass through `grad_slice`
  and the slab sum).

## Out of scope

- The matrices `B`, `A^grad`, `A^ibp`, `Q`, the three `α` solvers (grad, ibp, vm), the
  residual `R` and the improved estimator: next spec.
- The sampler (HMC, Metropolis, any generation of configurations) and the Stein identity
  `⟨ℒψ_j⟩ = 0`, which is a statement about expectations, not tested here.
- `λ ≠ 0`: `stein_psi` is not tested for it (the tests always pass `λ = 0`), and no
  behaviour is specified, although (★) holds for any `λ`.
- Real (non-synthetic) configurations, and ADerrors / `uwreal` (not involved).
- Bases other than level 1 (`M ≠ T`), and the 4d lattice.
- Input validation beyond the list in the Interface (no finiteness check, no range check on
  `κ`, `λ`).
- Performance: no in-place, threaded or allocation-free versions.
- Changes to existing code: specs 001 and 002 and their functions stay as they are.

## Open questions

(none)
