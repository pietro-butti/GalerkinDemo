# 001 – correlator-and-errors

## Interface

All functions are pure on arrays (no file I/O), except where stated below.
Data layout for this spec: `z` has size `(T, L, N)` = (time slab `s`, spatial site `x`,
configuration `n`); the configuration index is last. `Π` and `C` have size `(T, N)`.

```julia
slab_sums(z::AbstractArray{<:Real,3}) -> Matrix{Float64}
    # (T, N); Π[s,n] = Σ_x z[s,x,n]

correlator_samples(Π::AbstractMatrix{<:Real}) -> Matrix{Float64}
    # (T, N); row t+1 holds C(t) for t = 0..T-1, periodic in time

correlator(z::AbstractArray{<:Real,3}, id::String) -> Vector{uwreal}
    # length T; element t+1 is C(t); built as
    # uwreal(correlator_samples(slab_sums(z))[t+1, :], id)
    # same `id` for all t, so the C(t) are correlated with each other

correlator_stats(C::Vector{uwreal}, id::String) -> NamedTuple
    # calls uwerr on every element of C (mutates C; the only impure function here)
    # returns (t, value, err, taui, window):
    #   t::UnitRange{Int} = 0:T-1, the others Vector{Float64} / Vector{Int} of length T

plot_correlator(stats::NamedTuple) -> Plots.Plot
    # C(t) vs t with error bars, log y-axis; returned, never saved
```

Assumptions: one replica, one contiguous MC history (no gaps), `uwerr(a)` with no
extra argument, `id` is required, no deprecated `Int64` ids.

Input validation (throws, never returns NaN silently):
- `z` with `T < 1`, `L < 1` or `N < 2` → `ArgumentError`
- non-finite entries in `z` → `ArgumentError`
- `C` empty in `correlator_stats` → `ArgumentError`
- `stats` in `plot_correlator` lacking any of `t, value, err` or with unequal lengths
  → `ArgumentError` / `DimensionMismatch`

## Formulas

Indices: time `s, τ ∈ {0..T-1}` (array index `s+1`), site `x ∈ {0..L-1}`, configuration
`n ∈ {1..N}`. Time is periodic: `τ+t` is taken modulo `T`.

**Slab sum** (`slab_sums`):

    Π_n(s) = Σ_{x=0}^{L-1} z_n(s, x)

**Two-point function, per configuration** (`correlator_samples`), for `t = 0..T-1`:

    c_n(t) = (1/T) Σ_{τ=0}^{T-1} Π_n(τ) · Π_n((τ+t) mod T)

Consequence: `c_n(t) = c_n(T-t)` for `t = 1..T-1`.

**Ensemble average** (`value` of the `uwreal`):

    C(t) = (1/N) Σ_{n=1}^{N} c_n(t)

**Error analysis** (`correlator_stats`). For each `t`, `uwerr(C[t+1])` is called with no
extra argument. `value`, `err`, `taui` and `window` are returned exactly as ADerrors
reports them; this spec defines no formula for them (see Out of scope). If `uwerr`
throws for some `t` (e.g. `ArgumentError` when no window can be determined),
`correlator_stats` lets it propagate (no catching, no `NaN`).

The `C(t)` for different `t` share one `id`, hence one MC history: their covariance is
that of the ADerrors error propagation. No covariance matrix is exposed by this spec.

**Plot** (`plot_correlator`): points `(t, value[t])`, vertical error bars `± err[t]`,
y-axis logarithmic. Points with `value ≤ 0` cannot be drawn on a log axis and are omitted
(not clamped). Axis labels: `t`, `C(t)`.

## Acceptance criteria

Tolerance convention: "exact" means `isapprox(a, b; rtol=1e-14, atol=1e-14)` (all inputs
are small integer-valued or hand-computable). Every test that builds a `uwreal` uses its
own unique `id` string, because ADerrors keeps ids in a global registry.

- **AC-1: slab sums.** `z` of size `(2,3,2)` with `z[:,:,1] = [1 2 3; 4 5 6]` and
  `z[:,:,2] = [-1 0 1; 2 2 2]`. `slab_sums(z) == [6 0; 15 6]` (size `(2,2)`), exact.
- **AC-2: rolled correlator by hand.** `Π = [1 1; 2 0; 3 0; 4 0]` (T=4, N=2).
  `correlator_samples(Π)` is `[7.5 0.25; 6 0; 5.5 0; 6 0]`, exact. Also `T=1`, `Π=[3.0;;]`
  gives `[9.0;;]`.
- **AC-3: periodic symmetry.** `z = randn(MersenneTwister(1234), 8, 3, 10)`.
  For `t = 1..7`, row `t+1` of `correlator_samples(slab_sums(z))` equals row `8-t+1`,
  `rtol=1e-12`.
- **AC-4: ensemble mean.** `z` of size `(4,1,2)` with `z[:,1,:] = Π` from AC-2.
  `value.(correlator(z, id))` is `[3.875, 3.0, 2.75, 3.0]`, exact; length is `T = 4`.
- **AC-5: shared id correlates the t's.** With `z` from AC-3 and one `id`, after `uwerr`
  on `D = C[2] - C[8]` (t=1 and t=7, equal sample by sample): `value(D)` and `err(D)` are
  `0` with `atol=1e-12`. (With independent ids `err(D)` would be nonzero.)
- **AC-6: constant data.** `z = ones(8,4,50)`. After `correlator_stats`: `value == 16` for
  all t (exact), `err == 0` (`atol=1e-14`).
- **AC-7: stats structure.** For `z` from AC-3 with N=2000 instead of 10 (seed 1234):
  `stats.t == 0:7`; `value`, `err`, `taui`, `window` have length 8 (`taui`, `window` values are not checked); `stats.value`
  equals `value.(C)` and `stats.err` equals `err.(C)` exactly; all `err ≥ 0`.
- **AC-8: iid error scale.** Same data as AC-7. For each t, `stats.err[t+1]` agrees with
  `std(c_n(t)) / √N` to `rtol=0.2`. (Expected for independent samples, where τ_int ≈ 1/2.)
- **AC-9: no mutation, no I/O.** `z` and `Π` are unchanged (`==` to a copy taken before)
  after `slab_sums`, `correlator_samples`, `correlator`. `plot_correlator` creates no file
  in the working directory.
- **AC-10: invalid input.** Each throws the stated type:
  `z` with `T=0` or `L=0` or `N=1` → `ArgumentError`; `z` containing `NaN` or `Inf` →
  `ArgumentError`; `correlator_stats(uwreal[], id)` → `ArgumentError`;
  `plot_correlator` with `err` missing → `ArgumentError`;
  `value` and `err` of different length → `DimensionMismatch`.
- **AC-11: plot.** `plot_correlator` returns a `Plots.Plot` with log y-scale.
  For `stats = (t=0:2, value=[1.0,-1.0,2.0], err=[0.1,0.1,0.1])` exactly two points
  are plotted (t=0 and t=2).

## Out of scope

- File I/O of any kind (reading configurations, writing results or figures, BDIO).
- Folding `C(t)` to `t ≤ T/2`, effective mass, τ_int(t) plots, fits.
- Vacuum or disconnected subtraction; normalisation of `Π` by `L`.
- Details of the Γ-method analysis: autocovariance, τ_int, windowing criterion, bias
  correction, and any check of `taui` or `window` values.
- Multiple replicas, gapped histories, binning, non-default `uwerr` windowing (`wpm`).
- Covariance matrix between different `t`, and any derived quantity of the `C(t)`.
- Legacy `Int64` ensemble ids.
- Catching or working around ADerrors windowing failures.

## Open questions

(none)
