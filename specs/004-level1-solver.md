# 004 – level1-solver

## Goal

For the free scalar theory (λ = 0 of spec 002) on a periodic T × L lattice with the level-1
basis of spec 003, build the sample matrices B, A^grad, A^ibp, Q and solve for the Galerkin
coefficients α with one routine, selected by a symbol `:grad`, `:ibp` or `:vm`. Also provide
the pointwise residual R(z; α). The tests compare the three α and the residuals with the
exact free-theory solution and the sample identities derived below, which hold on any
sample with N > T, so random (non-Gaussian) data suffices.

## Interface

Layout as in spec 003: a single configuration `z` has size `(T, L)`, a batch has size
`(T, L, N)`, configuration index last. `M` is the number of basis functions (level 1
requires `M == T`), `p = (λ, κ)` is the argument tuple of `grad_slice` (spec 002); only
`λ = 0` is specified and tested. Index `j = 0..T-1` is Julia index `j+1`.

```julia
galerkin_matrices(z::AbstractArray{<:Real,3}, M, p)
    -> NamedTuple (B, Agrad, Aibp, Q), each Matrix of size (M, M)
    # B     = cov(ψ_i, ψ_j)                  sample, 1/(N-1)
    # Agrad = mean_n (∇ψ_i·∇ψ_j)             plain average, not a covariance
    # Aibp  = -cov(ψ_i, ℒψ_j)                sample, 1/(N-1); not symmetrised
    # Q     = cov(ℒψ_i, ℒψ_j)                sample, 1/(N-1)

solve_alpha(z::AbstractArray{<:Real,3}, M, p, kind::Symbol) -> Vector, size (M,)
    # kind ∈ (:grad, :ibp, :vm); source s = e₀ fixed
    # :grad  Agrad \ (B*s)
    # :ibp   Aibp  \ (B*s)
    # :vm    Q     \ (Aibp' * s)

stein_residual(z::AbstractMatrix{<:Real}, M, p, α)  -> Real
    # R(z; α) = ψ₀(z) + Σ_j α_j ℒψ_j(z)
stein_residual(z::AbstractArray{<:Real,3}, M, p, α) -> Vector, size (N,)
```

`galerkin_matrices` and `solve_alpha` have only the batch method (they need `N`); a
rank-2 `z` throws `ArgumentError`. The `(T, L, N)` method of `stein_residual` is
`stack`/`map` of the `(T, L)` method over `eachslice(z; dims=3)`. The matrices are built
from `basis`, `basis_grad` and `stein_psi` (spec 003) and `Statistics.cov`; no formula of
003 is copied. Linear systems are solved with `\`, never `inv`.

All methods are pure: no mutation of `z`, no I/O. Element types follow the inputs; nothing
is forced to `Float64`, so that ForwardDiff dual numbers pass through.
`galerkin_matrices`, `solve_alpha`, `stein_residual` are exported from
`src/GalerkinDemo.jl`; the code lives in `src/GalerkinDemo_level1.jl`.

Input validation:
- `ArgumentError`: rank of `z` wrong (not 3 for `galerkin_matrices` and `solve_alpha`; not
  2 or 3 for `stein_residual`); `T < 1`, `L < 1` or `N < 1`; `M ≠ T`; `N ≤ T` (B̂ singular,
  `galerkin_matrices` and `solve_alpha` only); `kind` not one of `:grad`, `:ibp`, `:vm`.
- `DimensionMismatch`: `length(α) ≠ M` in `stein_residual`.
- No NaN is returned silently; no finiteness check.

## Formulas

Indices: time `t, i, j ∈ {0..T-1}` (array index +1), site `x ∈ {0..L-1}`, configuration
`n ∈ {1..N}`. Time indices are taken modulo `T`. `δ_ab` is the Kronecker delta; `z_n` is
`z[:,:,n]`. The force `F = grad_slice(λ, κ)(z)` is that of spec 002.

**Basis and Stein operator** (spec 003):

    ψ_j(z) = Σ_x z(j,x),      ∂ψ_j/∂z(t,x) = δ_tj,      ∇ψ_i·∇ψ_j = Σ_{t,x} δ_ti δ_tj = L δ_ij,
    ℒψ_j(z) = −Σ_x F(j,x),    λ = 0.

**Sample covariance.** For functions `f, g` of `z`, with `f̄ = (1/N) Σ_n f(z_n)`:

    cov(f, g) = (1/(N−1)) Σ_n (f(z_n) − f̄)(g(z_n) − ḡ).

**Sample matrices** (`i, j = 0..T-1`):

    B_ij     = cov(ψ_i, ψ_j)
    Agrad_ij = (1/N) Σ_n ∇ψ_i·∇ψ_j (z_n)
    Aibp_ij  = − cov(ψ_i, ℒψ_j)
    Q_ij     = cov(ℒψ_i, ℒψ_j)

**Source.** `s = e₀`, i.e. `s_j = δ_j0`. In the free theory ⟨Π⟩ = 0, so the source
`Δ = Π(0) = ψ₀` lies in the span.

**Solvers:**

    :grad   Agrad · α = B s
    :ibp    Aibp  · α = B s
    :vm     Q · α = Aibp' · s            (' is the transpose)

**Residual**, pointwise in `z`, for a coefficient vector `α`:

    R(z; α) = ψ₀(z) + Σ_j α_j ℒψ_j(z).

**Free-theory reference.** `K0` is the `T × T` periodic matrix

    K0_ij = (2 − 4κ) δ_ij − 2κ (δ_{i,j+1} + δ_{i,j−1}),      G0 = K0⁻¹.

For `T ≥ 3`, `K0` is symmetric. By spec 003 (★★), `ℒψ_j = −Σ_i K0_ij ψ_i` pointwise, for
any `z`.

**Consequences, valid on any sample with `N > T`** (pointwise; no Gaussianity needed):

    Agrad = L I
    Aibp  = B K0,        Q = K0 B K0
    α_ibp = α_vm = G0[:,0] =: α*           (K0 α* = e₀)
    α_grad = B[:,0] / L
    R(z; α) = Σ_i (e₀ − K0 α)_i ψ_i(z)
    ⇒ R(z; α_ibp) = R(z; α_vm) = 0 for every z,
      R(z; α_grad) = −Σ_i (K0 α_grad − e₀)_i ψ_i(z).

Derivation: `Aibp_ij = −cov(ψ_i, −Σ_k K0_kj ψ_k) = (B K0)_ij`; `Q = K0ᵀ B K0 = K0 B K0`.
`:ibp` gives `B K0 α = B e₀`, and B is invertible, so `K0 α = e₀`. `:vm` gives
`K0 B K0 α = K0 B e₀`, so again `K0 α = e₀`. `:grad` gives `L α = B e₀`.

## Acceptance criteria

Common setup (unless an AC says otherwise): `T = 4`, `L = 3`, `M = T`, `κ = 0.1`, `λ = 0`,
`p = (0.0, 0.1)`, `N = 20` (`N ≠ T, L`, and `N > T`). Random batch:
`z = randn(Random.MersenneTwister(1234), 4, 3, 20)`. `K0` is the `4 × 4` periodic matrix
of Formulas, built by a helper in the test file; `G0 = K0 \ I`. "Exact" means
`isapprox(a, b; rtol=1e-14, atol=1e-14)`; "to 1e-12" means `rtol=1e-12, atol=1e-12`.
Each AC is one `@testset` "AC-<n>: <short title>" and also checks `size`. The tests use the
names of the Interface, `basis`, `basis_grad`, `stein_psi` (spec 003), `Statistics.cov`,
`ForwardDiff` and `LinearAlgebra`. Test helpers live in the test file, not in `src/`.

- **AC-1: hand anchors.** Formulas used: definitions above; for a constant field `z ≡ a`,
  `ψ_i = 3a` and `ℒψ_j = −3 (2 − 8κ) a = −3.6 a`.
  (a) `z[:,:,n] = fill(a_n, 4, 3)` with `a = [-2,-1,0,1,2]` (`N = 5 > T`, `var(a) = 2.5`):
  `galerkin_matrices(z, 4, p)` gives `B = 22.5·ones(4,4)`, `Aibp = 27·ones(4,4)`,
  `Q = 32.4·ones(4,4)`, `Agrad = 3·I`. Exact.
  (b) `z = fill(2.0, 4, 3)`: `stein_residual(z, 4, p, [1,0,0,0]) = 6 − 7.2 = −1.2` and
  `stein_residual(z, 4, p, ones(4)) = 6 − 28.8 = −22.8`. Exact.
- **AC-2: sample identities of the matrices.** Random `z`. Formulas used: Sample matrices,
  Consequences. With `ψ = basis(z,4)` and `Lψ = stein_psi(z,4,p)` (both `(4,20)`), to 1e-12:
  `B == cov(ψ')`, `Aibp == −cov(ψ', Lψ')`, `Q == cov(Lψ')`; exact: `Agrad == 3·I`.
  Then `Aibp == B*K0` and `Q == K0*B*K0`, to 1e-12. All four have size `(4,4)`.
- **AC-3: three solvers.** Random `z`. Formulas used: Solvers, Consequences. With
  `α* = G0[:,1]`, to 1e-12: `solve_alpha(z,4,p,:ibp) == α*`, `solve_alpha(z,4,p,:vm) == α*`,
  `solve_alpha(z,4,p,:grad) == B[:,1]/3` with `B = cov(ψ')`. Also
  `norm(solve_alpha(z,4,p,:grad) − α*) > 1e-2` (observed 0.37). Size `(4,)`.
- **AC-4: residuals and cross-fitting.** Formula used: `R(z;α) = Σ_i (e₀ − K0 α)_i ψ_i(z)`.
  Fit α on `zA` (the random batch) and apply on an independent `zB = randn(MersenneTwister(5678), 4, 3, 20)`.
  To 1e-12 (atol only for the zeros): `stein_residual(zB,4,p,α_ibp)` and `...α_vm` equal
  `zeros(20)`; `stein_residual(zB,4,p,α_grad) == −(K0*α_grad − e₀)' * basis(zB,4)`
  (a length-20 vector). The same holds on `zA`. Size `(20,)`.
- **AC-5: batch consistency, shapes, independence.** Formula used: Interface (batch =
  single-configuration method stacked on the last dimension).
  `stein_residual(z,4,p,α)[n] == stein_residual(z[:,:,n],4,p,α)` for every `n`, exact
  (`==`), with `α = G0[:,1]`. Replacing `z[:,:,2]` by other random values leaves the
  entries `n ≠ 2` unchanged (`==`).
- **AC-6: invalid input.** Formula used: Interface, Input validation. `ArgumentError` for:
  `galerkin_matrices` and `solve_alpha` (`:ibp`) with `randn(4,3)` and `randn(12)`;
  `stein_residual` with `randn(12)` and `randn(4,3,5,2)`; all three with `M = 3` and
  `M = 5` on size `(4,3,20)`, and with `zeros(0,3,20)`, `zeros(4,0,20)`, `zeros(4,3,0)`;
  `galerkin_matrices` and `solve_alpha` with `randn(4,3,4)` (`N = T`); `solve_alpha` with
  `kind = :foo`. `DimensionMismatch` for `stein_residual(z, 4, p, ones(3))`.
- **AC-7: no mutation, element type, dual numbers.** (a) `z` equals a copy taken before
  the calls (`==`) after all calls. (b) With `z32 = Float32.(z)`, `p32 = (0f0, 0.1f0)` and
  `α32 = Float32[1,0,0,0]`: the four matrices, `solve_alpha` for each `kind`, and
  `stein_residual` have `eltype == Float32`. (c) `R` is linear in `z`, so
  `ForwardDiff.derivative(c -> sum(stein_residual(c .* z, 4, p, α)), 1.0)` equals
  `sum(stein_residual(z, 4, p, α))` to 1e-12, with `α = G0[:,1]`.

## Out of scope

- The improved estimator `Â_t = R Π(t) + L α_t`, any function that splits a sample into
  offline/online parts (AC-4 does the split by hand), and any use of ADerrors.
- The sampler (HMC, Metropolis, Gaussian draws) and everything that needs statistics over
  many samples: the prediction `E[Var R_grad/Var Δ] ≃ (T+1)/N`, the population variance of
  `R`, and the Stein identity `⟨ℒψ_j⟩ = 0`.
- `λ ≠ 0`; a general source `s` (only `s = e₀`); bases other than level 1; the 4d lattice.
- Rank-deficient or ill-conditioned B̂ with `N > T`: no check beyond `N ≤ T`.
- Input validation beyond the list in the Interface (no finiteness check, no range check
  on `κ`, `λ`).
- Performance: no in-place, threaded or allocation-free versions.
- Changes to existing code: specs 001–003 and their functions stay as they are.

## Open questions

(none)
