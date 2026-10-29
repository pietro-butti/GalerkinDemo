using Test
using Random
using Statistics
using GalerkinDemo
using ADerrors
using Plots
using JLD2

import ADerrors.err

## Load configurations (interacting theory)

    ENS    = "2d_l0.1727157_k0.2682627_L_32_8"
    ROOT   = "/Users/pietro/code/data_analysis/sadflow/"
    data    = load(joinpath(ROOT,"sadflow","LCP364",ENS,"$ENS.jld2"))

    z_samples = data["ϕ"][:,:,1,1:100:end]

    λ, κ = 0.1727157, 0.2682627
    p    = (λ, κ)
    T, L, N = size(z_samples)
    M    = T
    @show T, L, N

## Baseline correlator

    uwc = correlator(z_samples, ENS)
    (t,v,e) = plottable(uwc; tol=1e-3)

    scatter(t,v; yerror=e, yscale=:log10, ylims=(1e-3,Inf))

## Level-1 Galerkin: α, residual and improved correlator for the three choices

    kinds = (:grad, :ibp, :vm)

    res = map(kinds) do kind
        α = solve_alpha(z_samples, M, p, kind)
        R = rolled_residual(z_samples, M, p, α)                       # (T, N)
        Â = improved_correlator(z_samples, M, p, α, ENS)
        (; kind, α, R, Â)
    end

    # residual summary: mean(R) ~ 0 (unbiasedness), std(R) drives the variance
    for r in res
        println(rpad(r.kind, 6), " mean(R) = ", round(mean(r.R); sigdigits=3),
                                "  std(R) = ", round(std(r.R);  sigdigits=3))
    end
    println("std(Π(0)) (no improvement) = ", round(std(basis(z_samples, M)[1, :]); sigdigits=3))

## Variance reduction factor (err_base / err_improved)²

    uwc = correlator(z_samples, ENS)
    foreach(uwerr, uwc)
    err_base = Float64[err(c) for c in uwc]

    vrf = map(res) do r
        foreach(uwerr, r.Â)
        (err_base ./ Float64[err(c) for c in r.Â]) .^ 2
    end

## Plots: per method, baseline + improved (left) and variance reduction factor (right)

    panels = map(zip(res, vrf)) do (r, f)
        base = plottable(correlator(z_samples, ENS); tol=1e-3)
        impr = plottable(r.Â; tol=1e-3)

        pc = scatter(base.t, base.value; yerror=base.err, yscale=:log10, ylims=(1e-3,Inf),
                     label="baseline", title=":$(r.kind)", xlabel="t", ylabel="C(t)",)
        scatter!(pc, impr.t .+ 0.15, impr.value; yerror=impr.err, label="improved")

        pv = plot(0:T-1, f; marker=:o, label=false, xlabel="t", ylabel="(σ_base/σ_impr)²",
                  title="variance reduction :$(r.kind)")
        hline!(pv, [1]; ls=:dash, color=:gray, label=false)
        (pc, pv)
    end

    plot(vcat(collect.(panels)...)...; layout=(3,2), size=(1000,1100))

##
