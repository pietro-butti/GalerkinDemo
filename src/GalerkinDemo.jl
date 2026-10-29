module GalerkinDemo

using ADerrors: uwreal, uwerr, value, err, taui, window
import Plots
using Statistics: cov

greet() = print("Hello World!")

export slab_sums, correlator_samples, correlator, correlator_stats, plottable, plot_correlator
export nn, action_slice, grad_slice, action, grad_S

# Throws unless `z` is a (T, L, N) array with T, L >= 1, N >= 3 and finite entries.
function _check_config_array(z::AbstractArray{<:Real,3})
    T, L, N = size(z)
    (T >= 1 && L >= 1) ||
        throw(ArgumentError("z must have at least one time slab and one site, got size $(size(z))"))
    N >= 3 ||
        throw(ArgumentError("z must have at least 3 configurations, got N = $N"))
    all(isfinite, z) ||
        throw(ArgumentError("z contains non-finite entries"))
    return nothing
end

"""
    slab_sums(z::AbstractArray{<:Real,3}) -> Matrix{Float64}

Slab sums of configurations `z` (time slab `s`, site `x`, configuration `n`; array index
`s+1`, `x+1`, `n`):

    Π_n(s) = Σ_{x=0}^{L-1} z_n(s, x)

Returns a `(T, N)` matrix, element `[s+1, n]` holding `Π_n(s)`.
"""
function slab_sums(z::AbstractArray{<:Real,3})
    _check_config_array(z)
    return Matrix{Float64}(dropdims(sum(z; dims=2); dims=2))
end

"""
    correlator_samples(Π::AbstractMatrix{<:Real}) -> Matrix{Float64}

Zero-momentum two-point function of each configuration, averaged over all periodic
rollings of the slabs. For `t = 0..T-1` (row `t+1` of the result):

    c_n(t) = (1/T) Σ_{τ=0}^{T-1} Π_n(τ) Π_n((τ + t) mod T)

`Π` is the `(T, N)` matrix of slab sums; the result has the same size.
"""
function correlator_samples(Π::AbstractMatrix{<:Real})
    T, N = size(Π)
    c = zeros(Float64, T, N)
    for n in 1:N, t in 0:T-1
        acc = 0.0
        for τ in 0:T-1
            acc += Π[τ+1, n] * Π[mod(τ + t, T)+1, n]
        end
        c[t+1, n] = acc / T
    end
    return c
end

"""
    correlator(z::AbstractArray{<:Real,3}, id::String) -> Vector{uwreal}

Two-point function as one `uwreal` per time separation. Element `t+1` is

    C(t) = (1/N) Σ_{n=1}^{N} c_n(t),   c_n(t) = (1/T) Σ_τ Π_n(τ) Π_n((τ + t) mod T),
    Π_n(s) = Σ_x z[s, x, n]

built from the samples `c_n(t)` with ensemble name `id`. The same `id` is used for all
`t`, so the `C(t)` are correlated with each other. `z` has size `(T, L, N)`.
"""
function correlator(z::AbstractArray{<:Real,3}, id::String)
    c = correlator_samples(slab_sums(z))
    return [uwreal(c[t, :], id) for t in 1:size(c, 1)]
end

"""
    correlator_stats(C::Vector{uwreal}, id::String) -> NamedTuple

Runs `uwerr` (default settings) on every element of `C`, which mutates `C`. Returns
`(t, value, err, taui, window)` where `t = 0:length(C)-1` and the other fields hold
`value(C[t+1])`, `err(C[t+1])`, and the integrated autocorrelation time and summation
window of ensemble `id`, as reported by ADerrors.
"""
function correlator_stats(C::Vector{uwreal}, id::String)
    isempty(C) && throw(ArgumentError("C is empty"))
    foreach(uwerr, C)
    return (t = 0:length(C)-1,
            value = Float64[value(c) for c in C],
            err = Float64[err(c) for c in C],
            taui = Float64[taui(c, id) for c in C],
            window = Int[window(c, id) for c in C])
end

"""
    plottable(uwcorr::Vector{uwreal}; tol=1e-10) -> NamedTuple

Runs `uwerr` (default settings) on every element of `uwcorr`, which mutates it. Returns
`(t, value, err)` restricted to the points whose lowest error bar is above `tol`:

    value(C(t)) - err(C(t)) > tol

with `t = index - 1` (`Vector{Int}`) and `value`, `err` as `Vector{Float64}`, in the
original order. The result is empty if no point qualifies.
"""
function plottable(uwcorr::Vector{uwreal}; tol=1e-10)
    isempty(uwcorr) && throw(ArgumentError("uwcorr is empty"))
    foreach(uwerr, uwcorr)
    v = Float64[value(c) for c in uwcorr]
    e = Float64[err(c) for c in uwcorr]
    keep = findall(>(tol), v .- e)
    return (t = keep .- 1, value = v[keep], err = e[keep])
end

"""
    plot_correlator(stats::NamedTuple) -> Plots.Plot

Plot of `C(t)` against `t` with vertical error bars `± err` on a logarithmic y-axis.
`stats` needs fields `t`, `value`, `err` of equal length. Points with `value ≤ 0`
cannot be shown on a log axis and are omitted. The plot is returned, never saved.
"""
function plot_correlator(stats::NamedTuple)
    for k in (:t, :value, :err)
        haskey(stats, k) || throw(ArgumentError("stats lacks field $k"))
    end
    length(stats.t) == length(stats.value) == length(stats.err) ||
        throw(DimensionMismatch("stats.t, stats.value, stats.err must have equal length"))
    keep = findall(>(0), stats.value)
    isempty(keep) && throw(ArgumentError("stats.value has no positive entries to plot"))
    return Plots.scatter(collect(stats.t)[keep], stats.value[keep];
                         yerror = stats.err[keep], yscale = :log10,
                         xlabel = "t", ylabel = "C(t)", legend = false)
end

"""
    nn(ϕ, s)

Sum of `ϕ` circularly shifted by `s` along each dimension (2d and 4d arrays):

    nn(ϕ, s)(x) = Σ_μ ϕ(x + s e_μ)
"""
nn(ϕ::AbstractArray{T,2}, s) where T =
    circshift(ϕ,(s,0)) .+
    circshift(ϕ,(0,s))
nn(ϕ::AbstractArray{T,4}, s) where T =
    circshift(ϕ,(s,0,0,0)) .+
    circshift(ϕ,(0,s,0,0)) .+
    circshift(ϕ,(0,0,s,0)) .+
    circshift(ϕ,(0,0,0,s))

"""
    action_slice(λ, κ) -> (ϕ -> S)

Action of one `(T, L)` configuration on a periodic lattice:

    S(ϕ) = Σ_{s,x} [ ϕ² + λ (ϕ² - 1)² - 2κ ϕ(s,x) (ϕ(s-1,x) + ϕ(s,x-1)) ]
"""
action_slice(λ,κ) = ϕ ->
    ϕ.^2 .+ λ .* (ϕ.^2 .- 1).^2 .+ -2κ .* ϕ .* nn(ϕ,-1) |> sum

"""
    grad_slice(λ, κ) -> (ϕ -> ∂S/∂ϕ)

Gradient of [`action_slice`](@ref), same size as `ϕ`:

    ∂S/∂ϕ(s,x) = 2ϕ + 4λ (ϕ² - 1) ϕ
                  - 2κ (ϕ(s-1,x) + ϕ(s,x-1) + ϕ(s+1,x) + ϕ(s,x+1))
"""
grad_slice(λ,κ) = ϕ ->
    2ϕ .+ 4λ .* (ϕ.^2 .- 1) .* ϕ .+ -2κ .* (nn(ϕ,-1) .+ nn(ϕ,1))

"""
    action(λ, κ) -> (ϕ::AbstractArray{<:Real,3} -> Vector)

Action of every configuration of a `(T, L, N)` batch; element `n` of the result is

    S(ϕ[:,:,n]) = action_slice(λ, κ)(ϕ[:,:,n])
"""
action(λ,κ) = ϕ::AbstractArray{<:Real,3} ->
    map(action_slice(λ,κ), eachslice(ϕ; dims=3))

"""
    grad_S(λ, κ) -> (ϕ::AbstractArray{<:Real,3} -> Array{·,3})

Gradient of [`action`](@ref) for a `(T, L, N)` batch, same size as `ϕ`; slice `n` is

    ∂S/∂ϕ evaluated on ϕ[:,:,n] = grad_slice(λ, κ)(ϕ[:,:,n])
"""
grad_S(λ,κ) = ϕ::AbstractArray{<:Real,3} ->
    stack(map(grad_slice(λ,κ), eachslice(ϕ; dims=3)))

include("GalerkinDemo_level1.jl")

end # module GalerkinDemo
