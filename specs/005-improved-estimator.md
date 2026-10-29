# 005 – improved-estimator

## Goal

For the free scalar theory (λ = 0 of spec 002) on a periodic T × L lattice with the level-1
basis (spec 003) and coefficients α from spec 004, build the improved correlator
`Â_t = R Π(t) + f_t` (GALERKIN.md §1.6) and the control variate `CV_t = Â_t − U_t`, the piece
that is added to the standard estimator. Improved correlator and control variate are
returned as `uwreal`s with the same ensemble `id` as `correlator` (spec 001), so that
`C[t] + CV[t]` is correlated correctly by ADerrors and agrees with the improved correlator.
The tests check, on free-theory Gaussian samples, the Stein identity, the vanishing of the
residual for the exact α, and the equality of `C + CV` and the improved correlator.

## Interface

Layout as in specs 001–004: a batch `z` has size `(T, L, N)`, configuration index last;
`M == T`; `p = (λ, κ)` as in spec 004 (only `λ = 0` is specified and tested); `α` has
length `M` and is **passed in** (fitted elsewhere, e.g. `solve_alpha`; no cross-fitting or
fitting happens in this spec). Index `t, τ, j = 0..T-1` is Julia index `+1`.

```julia
shift_term(z::AbstractMatrix{<:Real}, M, α) -> Vector, size (T,)
    # f_t = Σ_j α_j Σ_x ∂ψ_j/∂z(t,x), from basis_grad (spec 003)

rolled_residual(z::AbstractArray{<:Real,3}, M, p, α) -> Matrix, size (T, N)
    # [τ+1, n] = R(z_n^τ; α), z_n^τ(t,x) = z_n((t+τ) mod T, x); stein_residual (spec 004)

improved_samples(z::AbstractArray{<:Real,3}, M, p, α)         -> Matrix, size (T, N)
control_variate_samples(z::AbstractArray{<:Real,3}, M, p, α)  -> Matrix, size (T, N)
    # per-configuration samples a_n(t), cv_n(t); row t+1 holds t, like correlator_samples

stein_violation(z::AbstractArray{<:Real,3}, M, p, α) -> Real
    # sample mean of ℒΦ = Σ_j α_j ℒψ_j; its population value is 0 (Stein identity)

ibp_violation(z::AbstractArray{<:Real,3}, M, p) -> Real
    # Frobenius distance ‖Agrad − Aibp‖_F between the two A matrices of spec 004

improved_correlator(z::AbstractArray{<:Real,3}, M, p, α, id::String) -> Vector{uwreal}
control_variate(z::AbstractArray{<:Real,3}, M, p, α, id::String)     -> Vector{uwreal}
    # length T; element t+1 is uwreal(<samples>[t+1, :], id), as in `correlator`
```

The residual `R` of the single source `Δ = Π(0)` is `stein_residual` of spec 004 and is
**not reimplemented**; `rolled_residual` calls it on the rolled configurations
(`circshift(z_n, (-τ, 0))`). `Π` comes from `basis`, `f_t` from `basis_grad`, the plain
estimator from `correlator_samples`/`slab_sums` (spec 001) only in the tests; no formula of
specs 001–004 is copied. Same `id` for all `t` and, in the intended use, the same `id` as
the `correlator(z, id)` it is combined with.

Element types: the sample functions and `shift_term` return
`promote_type(eltype(z), eltype(α), eltype of p...)`; the two `uwreal` functions convert the
samples to `Float64`. All functions are pure (no mutation, no I/O; `uwerr` is not called).
Exported from `src/GalerkinDemo.jl`, code in `src/GalerkinDemo_level1.jl`.

Input validation:
- `ArgumentError`: rank of `z` not 3 (`shift_term`: not 2); `T < 1`, `L < 1`, `N < 1`;
  `M ≠ T`. The two `uwreal` functions additionally require `N ≥ 3` and finite `z`
  (same rule as `correlator`, spec 001).
- `DimensionMismatch`: `length(α) ≠ M`.
- No NaN is returned silently.

## Formulas

Indices as in spec 004. `Π_n(s) = Σ_x z_n(s,x) = ψ_s(z_n)`; `z_n^τ` is `z_n` rolled by τ in
time. Since the action is invariant under time translations, the source at τ is handled by
rolling, so the single-source theory of GALERKIN.md §1.6 (`Δ = Π(0)`) is averaged over τ
exactly as `correlator_samples` averages `Π(τ)Π(τ+t)`.

**Residual of source τ** (via `stein_residual`):

    R_τ,n = R(z_n^τ; α) = ψ₀(z_n^τ) + Σ_j α_j ℒψ_j(z_n^τ) = Π_n(τ) + Σ_j α_j ℒψ_j(z_n^τ)

**Shift term.** For the level-1 basis `∂ψ_j/∂z(t,x) = δ_tj`, hence

    f_t = Σ_j α_j Σ_x δ_tj = L α_t.

**Standard estimator** (spec 001): `c_n(t) = (1/T) Σ_τ Π_n(τ) Π_n(τ+t)`.

**Improved estimator**, with `Π_n(τ+t) = Π_n((τ+t) mod T) = Π_t(z_n^τ)`:

    a_n(t) = f_t + (1/T) Σ_τ R_τ,n Π_n(τ+t)

**Control variate** (`Â − U`, with `R_τ,n − Π_n(τ) = Σ_j α_j ℒψ_j(z_n^τ)`):

    cv_n(t) = f_t + (1/T) Σ_τ (R_τ,n − Π_n(τ)) Π_n(τ+t)
    a_n(t) = c_n(t) + cv_n(t)        (identity, any z and any α)

**Ensemble values.** `Â(t) = (1/N) Σ_n a_n(t)`, `CV(t) = (1/N) Σ_n cv_n(t)`; the `uwreal`s
are `uwreal(a[t+1,:], id)` and `uwreal(cv[t+1,:], id)`.

**Metrics.** Diagnostics of how well a sample and an α respect the identities that make
the estimator unbiased.

    stein_violation(z; α) = (1/N) Σ_n Σ_j α_j ℒψ_j(z_n)         (population value ⟨ℒΦ⟩ = 0)
    ibp_violation(z)      = ‖Agrad − Aibp‖_F = sqrt(Σ_ij (Agrad_ij − Aibp_ij)²)

`Agrad` and `Aibp` are those of `galerkin_matrices` (spec 004, reused); they have the same
population value (`Cov(f, ℒg) = −⟨∇f·∇g⟩`), so `ibp_violation` measures the failure of
integration by parts on the sample (adjoint Stein identity). Both vanish only in the
population/limit, not on a finite sample; only deterministic cases are tested (AC-4).

**Unbiasedness** (population, any α): `⟨cv_n(t)⟩ = 0`, because `cv` is an average of
`𝒯(Π_t ∇Φ)` over τ and `⟨𝒯 g⟩ = 0`; equivalently `⟨ℒψ_j⟩ = 0`. Hence `⟨a_n(t)⟩ = C(t)`.

**Free-theory reference.** `K0`, `G0 = K0⁻¹` as in spec 004, `K0_ij = (2−4κ)δ_ij −
2κ(δ_{i,j+1}+δ_{i,j−1})`; `C(t) = L·G0[t,0]`. The free theory is the Gaussian
`z ~ N(0, K⁻¹)` with `K = 2I − 2κH` the full (TL × TL) matrix (`F = Kz`).

**Consequences.**
- `α* = G0[:,0]` gives `R_τ,n = 0` for every `τ, n` (spec 004: `R = Σ_i(e₀−K0α)_i ψ_i`, and
  rolling preserves the form), so `a_n(t) = f_t = L α*_t = C(t)` on every configuration:
  zero variance. `α_ibp = α_vm = α*` on any sample with `N > T`.
- For any α, `a_n(t) = c_n(t) + cv_n(t)`, so `Â(t) = C(t) + CV(t)` as numbers and, since
  both are linear in the same data with the same `id`, with the same ADerrors error.

## Acceptance criteria

Common setup: `T = 4`, `L = 3`, `M = T`, `κ = 0.1`, `p = (0.0, 0.1)`. "Exact" means
`isapprox(a, b; rtol=1e-14, atol=1e-14)`; "to 1e-12" means `rtol=1e-12, atol=1e-12`.
`K0` is the helper of spec 004. The test file provides the helper `gaussian_batch(rng, N)`
that returns `z` of size `(4,3,N)` with `z[:,:,n] ~ N(0, K⁻¹)`: build the full `K` by
applying `grad_slice(0.0, 0.1)` to the 12 unit configurations (`F = Kz` is linear),
`U = cholesky(Symmetric(K)).U`, `z[:,:,n] = reshape(U \ randn(rng, 12), 4, 3)`.
Random non-Gaussian batch: `zA = randn(MersenneTwister(1234), 4, 3, 20)`. Every test that
builds a `uwreal` uses its own unique `id` string. Each AC is one `@testset`
"AC-<n>: <short title>" and also checks `size`.

- **AC-1: hand anchors.** Formulas used: `f_t`, `a_n(t)`, `cv_n(t)`. For constant fields
  `z ≡ a` one has `Π = 3a`, `ℒψ_j = −3.6a`, `c = 9a²`. With `α = [1,0,0,0]`:
  `R = 3a − 3.6a = −0.6a` for every source, so `a_n(t) = −1.8a² + 3δ_t0` and
  `cv_n(t) = −10.8a² + 3δ_t0`. (a) `shift_term(zeros(4,3), 4, α) == [3,0,0,0]`.
  (b) `z[:,:,n] = fill(a_n,4,3)`, `a = [-2,-1,0,1,2]`: `rolled_residual(z,4,p,α)` has size
  `(4,5)` with every row `= −0.6a = [1.2,0.6,0,-0.6,-1.2]`; `improved_samples` has rows
  `[-4.2 1.2 3 1.2 -4.2]` (t=0) and `[-7.2 -1.8 0 -1.8 -7.2]` (t=1,2,3);
  `control_variate_samples` has rows `[-40.2 -7.8 3 -7.8 -40.2]` (t=0) and
  `[-43.2 -10.8 0 -10.8 -43.2]` (t=1,2,3). Exact.
- **AC-2: algebraic identities on any sample.** Formulas used: `a_n(t)`, `cv_n(t)`,
  `a = c + cv`. `zA`, `α = solve_alpha(zA, 4, p, :grad)`. To 1e-12:
  `improved_samples == correlator_samples(slab_sums(zA)) + control_variate_samples`;
  and `improved_samples` equals a direct loop in the test over `n, t, τ` using
  `stein_residual(circshift(zA[:,:,n], (-τ,0)), 4, p, α)`, `basis` and `shift_term`.
  The same for `α = ones(4)` (arbitrary α). Sizes `(4,20)`.
- **AC-3: exact α on free-theory samples.** Formulas used: Consequences.
  `zG = gaussian_batch(MersenneTwister(2024), 50)`, `α* = G0[:,1]`, and
  `α_vm = solve_alpha(zG, 4, p, :vm)`. To 1e-12 (atol only for zeros): for `α*` and for
  `α_vm`, `rolled_residual(zG,4,p,α) == zeros(4,50)`; every column of `improved_samples`
  equals `3 .* G0[:,1]` (`= C(t)`); every column of `control_variate_samples` equals
  `3 .* G0[:,1] − correlator_samples(slab_sums(zG))[:,n]`.
- **AC-4: metrics.** Formulas used: Metrics. (a) Stein violation, hand values: `z[:,:,n] =
  fill(a_n,4,3)`, `a = [1,2,3]`: `stein_violation(z,4,p,[1,0,0,0]) = −7.2` and
  `stein_violation(z,4,p,ones(4)) = −28.8`, exact. (b) On `zA`: equals
  `mean(stein_psi(zA,4,p)' * α)` and is linear in `α` (`v(α₁+2α₂) = v(α₁)+2v(α₂)`), to
  1e-12. (c) Antithetic batch `cat(zG, −zG; dims=3)` with `zG = gaussian_batch(MersenneTwister(2024), 50)`:
  `stein_violation` is `0` (`atol=1e-12`) for `α_grad` (fitted on `zA`), `α*` and `ones(4)`;
  this is the Stein identity on a symmetric sample (`ℒψ_j` is odd in `z`). (d) IBP violation,
  hand value: for `z` of (a)-type with `a = [-2,-1,0,1,2]`, `Agrad = 3I` and
  `Aibp = 27·ones(4,4)` (spec 004 AC-1), so `ibp_violation(z,4,p) = √11052`, exact.
  (e) On `zA`: `ibp_violation == norm(Agrad − Aibp)` (Frobenius) from `galerkin_matrices`,
  and `== norm(B*K0 − 3I)` (spec 004 consequence), to 1e-12; it is `> 0`. Both metrics
  return a scalar (`Real`), and `ibp_violation` throws `ArgumentError` on `randn(4,3)`,
  on `N = T` (`randn(4,3,4)`) and `M ≠ T`; `stein_violation` throws `ArgumentError` on
  rank-2 `z` and `M ≠ T`, `DimensionMismatch` for `length(α) = 3`.
- **AC-5: `C + CV` equals the improved correlator.** Formulas used: identity `a = c + cv`,
  Ensemble values. `z = gaussian_batch(MersenneTwister(99), 2000)`,
  `α = solve_alpha(gaussian_batch(MersenneTwister(7), 40), 4, p, :grad)`, one `id`.
  `C = correlator(z,id)`, `CV = control_variate(z,4,p,α,id)`, `Â = improved_correlator(z,4,p,α,id)`;
  all three have length 4 and eltype `uwreal`. `S = C .+ CV`. After `uwerr` on every `S[t]`
  and `Â[t]`: `value(S[t]) == value(Â[t])` with `rtol=1e-10`, and `err(S[t]) == err(Â[t])`
  with `rtol=1e-8, atol=1e-12`, for all `t`; `value(Â[t])` equals the row mean of
  `improved_samples` to 1e-12.
- **AC-6: exact α gives zero error.** Formulas used: Consequences. Same `z` as AC-5 (own
  `id`), `α* = G0[:,1]`. `Â = improved_correlator(z,4,p,α*,id)`: after `uwerr`,
  `value(Â[t]) == 3·G0[t,0]` with `rtol=1e-10` and `err(Â[t]) ≤ 1e-9`, while the plain
  `err(correlator(z, id)[t])` (after `uwerr`) is `> 1e-3`. Also `value(C[t] + CV[t])`
  equals `3·G0[t,0]` with `rtol=1e-10`.
- **AC-7: invalid input.** Formulas used: Interface, Input validation. `ArgumentError`:
  `rolled_residual`, `improved_samples`, `control_variate_samples` with `randn(4,3)` and
  `randn(12)`; `shift_term` with `randn(4,3,5)` and `randn(12)`; all functions with `M = 3`
  and `M = 5` on size `(4,3,20)`, and with `zeros(0,3,20)`, `zeros(4,0,20)`, `zeros(4,3,0)`;
  `improved_correlator` and `control_variate` with `N = 2` (`randn(4,3,2)`), and with a
  `NaN` entry in `z`. `DimensionMismatch` for `length(α) = 3` in all seven functions.
- **AC-8: no mutation, element type.** (a) `zA`, `α` equal copies (`==`) taken before the
  calls, after all calls. (b) With `Float32.(zA)`, `p32 = (0f0, 0.1f0)`, `α32 = Float32[1,0,0,0]`:
  `shift_term`, `rolled_residual`, `improved_samples`, `control_variate_samples` have
  `eltype == Float32`.

## Out of scope

- Statistical tests of the Stein identity or of unbiasedness (`⟨ℒψ_j⟩ = 0`, `⟨cv⟩ = 0`) on
  finite i.i.d. samples; a normalised/relative `ibp_violation`; a standard error of
  `stein_violation`.
- Fitting α or cross-fitting (the offline/online split); `α` is an argument.
- Predicted variance reduction of the grad solver (`(T+1)/N`), any claim that `err(Â) <
  err(C)` for `:grad`, and the population variance of `R`.
- `λ ≠ 0`; sources other than `e₀`-translated (all τ); bases other than level 1; 4d.
- Folding in `t`, plotting, effective masses, calling `uwerr` inside the library.
- Changes to existing code: specs 001–004 and their functions stay as they are.

## Notes

- Conflict resolved by decision: GALERKIN.md §1.6 has a single source `Δ = Π(0)`, while
  `correlator` (spec 001) averages over all τ. To make `C + CV` the same estimator as
  the improved correlator, `Â` and `CV` are averaged over τ in the same way (via rolling, which
  is exact for the translation-invariant action).
- `R` for the single source τ=0 is `stein_residual` from spec 004; no new residual formula.
- Risk: in AC-6 the improved samples are constant up to rounding (~1e-16); if `uwerr`
  throws on such data, report it, do not catch it in the library.

## Open questions

(none)
