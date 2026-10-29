# 002 – lambdaphi4

## Goal

Provide the action `S(ϕ)` and its gradient `∂S/∂ϕ` of the 2d lattice λ-φ⁴ model, as
functions of the couplings `(λ, κ)` that return closures acting on configurations. The
closures work on a whole batch `(T, L, N)` of configurations, by applying the single
configuration `(T, L)` routines to each slice. The point of this spec is that the batch
version is consistent with the single-slice one, and that the gradient is the gradient
of the action.

## Interface

Data layout: a batch `ϕ` has size `(T, L, N)` = (time `s`, site `x`, configuration `n`),
configuration index last (same as spec 001-correlator-and-errors).

Single configuration, `ϕ` of size `(T, L)`. Taken from the existing analysis code (see
Formulas); `_action` and `_grad_S` there are renamed `action_slice` and `grad_slice`:

```julia
nn(ϕ::AbstractArray{T,2}, s) where T         # nearest-neighbour sum, shift s
action_slice(λ, κ) -> (ϕ::AbstractArray{<:Real,2} -> Real)
grad_slice(λ, κ) -> (ϕ::AbstractArray{<:Real,2} -> Array{·,2})   # same size as ϕ
```

Batch (new, built from the two single-slice closures, no duplicated formulas):

```julia
action(λ, κ) -> (ϕ::AbstractArray{<:Real,3} -> Vector)
    # length N; element n is S(ϕ[:,:,n])
grad_S(λ, κ) -> (ϕ::AbstractArray{<:Real,3} -> Array{·,3})
    # size(ϕ); [:,:,n] is ∂S/∂ϕ evaluated on ϕ[:,:,n]
```

Element types follow the input (`Float64` in, `Float64` out; `ForwardDiff.Dual` in,
`Dual` out). Nothing is forced to `Float64`, so that automatic differentiation passes
through. Pure functions, no mutation of `ϕ`, no I/O.

All five names (`nn`, `action_slice`, `grad_slice`, `action`, `grad_S`) live in
`src/GalerkinDemo.jl` and are exported. Existing functions are not changed.

## Formulas

Verbatim from the existing code (copy, do not rewrite):

```julia
nn(ϕ::AbstractArray{T,2}, s) where T = 
    circshift(ϕ,(s,0)) .+ 
    circshift(ϕ,(0,s))
nn(ϕ::AbstractArray{T,4}, s) where T = 
    circshift(ϕ,(s,0,0,0)) .+ 
    circshift(ϕ,(0,s,0,0)) .+ 
    circshift(ϕ,(0,0,s,0)) .+ 
    circshift(ϕ,(0,0,0,s))

action_slice(λ,κ) = ϕ ->
    ϕ.^2 .+ λ .* (ϕ.^2 .- 1).^2 .+ -2κ .* ϕ .* nn(ϕ,-1) |> sum

grad_slice(λ,κ) = ϕ -> 
    2ϕ .+ 4λ .* (ϕ.^2 .- 1) .* ϕ .+ -2κ .* (nn(ϕ,-1) .+ nn(ϕ,1))
```

In symbols, for a configuration `ϕ(s,x)` on a periodic `T × L` lattice
(`s` = 1..T, `x` = 1..L, indices mod `T`, `L`):

    S(ϕ) = Σ_{s,x} [ ϕ² + λ (ϕ² - 1)² - 2κ ϕ(s,x) (ϕ(s-1,x) + ϕ(s,x-1)) ]

    ∂S/∂ϕ(s,x) = 2ϕ + 4λ (ϕ² - 1) ϕ
                  - 2κ (ϕ(s-1,x) + ϕ(s,x-1) + ϕ(s+1,x) + ϕ(s,x+1))

Batch:

    action(λ,κ)(ϕ)[n]        = action_slice(λ,κ)(ϕ[:,:,n])
    grad_S(λ,κ)(ϕ)[:,:,n]    = grad_slice(λ,κ)(ϕ[:,:,n])

Remark: the `nn` / `grad_slice` pair is consistent also for `T` or `L` equal to 1 or 2
(where `s+1 ≡ s-1`), so the small test sizes below are legitimate.

## Acceptance criteria

The single-slice formulas are already tested elsewhere and are not re-tested. All tests
use `Random.MersenneTwister(1234)`; `ϕ = randn(rng, 2, 3, 4)` (T=2, L=3, N=4),
`λ = 0.7`, `κ = 0.3`. "Exact" means `==`. The tests need ForwardDiff in
`test/Project.toml` (test dependency only, not a package dependency).

- **AC-1: batch consistency, action.** `S = action(λ,κ)(ϕ)` has `size(S) == (4,)` and
  `S[n] == action_slice(λ,κ)(ϕ[:,:,n])` for every `n`, exact.
- **AC-2: batch consistency, gradient.** `g = grad_S(λ,κ)(ϕ)` has `size(g) == size(ϕ)` and
  `g[:,:,n] == grad_slice(λ,κ)(ϕ[:,:,n])` for every `n`, exact.
- **AC-3: configurations are independent.** Replacing `ϕ[:,:,2]` by other random values
  leaves `S[n]` and `g[:,:,n]` unchanged for `n ≠ 2`, exact.
- **AC-4: gradient vs ForwardDiff.** `ForwardDiff.gradient(ψ -> sum(action(λ,κ)(ψ)), ϕ)`
  equals `grad_S(λ,κ)(ϕ)`, `rtol=1e-12, atol=1e-12`. (Sum over `n` is fine because the
  configurations do not interact.) Also per configuration: for each `n`,
  `ForwardDiff.gradient(ψ -> action_slice(λ,κ)(ψ), ϕ[:,:,n])` equals `g[:,:,n]`, same tolerance.
- **AC-5: gradient vs finite differences.** Central differences with `h = 1e-6` on every
  entry `i` of `ϕ`: `(Σ action(ϕ + h eᵢ) - Σ action(ϕ - h eᵢ)) / 2h` equals `g[i]`,
  `rtol=1e-6, atol=1e-8`.
- **AC-6: no mutation.** `ϕ` equals a copy taken before the calls (`==`), after both
  `action` and `grad_S`.

## Out of scope

- The 4d routines are copied with `nn` for completeness but not tested or used here.
- Any sampler (HMC, leapfrog, Metropolis), Hessian, or force normalisation.
- Observables, correlators (spec 001-correlator-and-errors), ADerrors (not involved).
- Input validation of any kind (no new `ArgumentError`s, no finiteness or size checks,
  no range checks on `λ`, `κ`); `_check_config_array` is not used.
- Performance: no in-place or threaded versions.

## Open questions

(none)
