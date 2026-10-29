using Test
using Random
using ForwardDiff
using GalerkinDemo

# Spec 002-lambdaphi4: batch action and gradient of the 2d λ-φ⁴ model.
_rng002() = Random.MersenneTwister(1234)
const _λ002 = 0.7
const _κ002 = 0.3

@testset "spec 002: lambdaphi4" begin
    @testset "AC-1: batch consistency, action" begin
        ϕ = randn(_rng002(), 2, 3, 4)
        S = action(_λ002, _κ002)(ϕ)
        @test size(S) == (4,)
        for n in 1:4
            @test S[n] == action_slice(_λ002, _κ002)(ϕ[:, :, n])
        end
    end

    @testset "AC-2: batch consistency, gradient" begin
        ϕ = randn(_rng002(), 2, 3, 4)
        g = grad_S(_λ002, _κ002)(ϕ)
        @test size(g) == size(ϕ)
        for n in 1:4
            @test g[:, :, n] == grad_slice(_λ002, _κ002)(ϕ[:, :, n])
        end
    end

    @testset "AC-3: configurations are independent" begin
        rng = _rng002()
        ϕ = randn(rng, 2, 3, 4)
        S = action(_λ002, _κ002)(ϕ)
        g = grad_S(_λ002, _κ002)(ϕ)

        ϕ2 = copy(ϕ)
        ϕ2[:, :, 2] = randn(rng, 2, 3)
        S2 = action(_λ002, _κ002)(ϕ2)
        g2 = grad_S(_λ002, _κ002)(ϕ2)

        for n in (1, 3, 4)
            @test S2[n] == S[n]
            @test g2[:, :, n] == g[:, :, n]
        end
    end

    @testset "AC-4: gradient vs ForwardDiff" begin
        ϕ = randn(_rng002(), 2, 3, 4)
        g = grad_S(_λ002, _κ002)(ϕ)

        gad = ForwardDiff.gradient(ψ -> sum(action(_λ002, _κ002)(ψ)), ϕ)
        @test isapprox(gad, g; rtol=1e-12, atol=1e-12)

        for n in 1:4
            gn = ForwardDiff.gradient(ψ -> action_slice(_λ002, _κ002)(ψ), ϕ[:, :, n])
            @test isapprox(gn, g[:, :, n]; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "AC-5: gradient vs finite differences" begin
        ϕ = randn(_rng002(), 2, 3, 4)
        g = grad_S(_λ002, _κ002)(ϕ)
        S = action(_λ002, _κ002)
        h = 1e-6
        for i in eachindex(ϕ)
            ϕp = copy(ϕ); ϕp[i] += h
            ϕm = copy(ϕ); ϕm[i] -= h
            fd = (sum(S(ϕp)) - sum(S(ϕm))) / 2h
            @test isapprox(fd, g[i]; rtol=1e-6, atol=1e-8)
        end
    end

    @testset "AC-6: no mutation" begin
        ϕ = randn(_rng002(), 2, 3, 4)
        ϕ0 = copy(ϕ)
        action(_λ002, _κ002)(ϕ)
        @test ϕ == ϕ0
        grad_S(_λ002, _κ002)(ϕ)
        @test ϕ == ϕ0
    end
end
