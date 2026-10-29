using Test
using Random
using ForwardDiff
using Statistics: cov
using LinearAlgebra
using GalerkinDemo

# Spec 004-level1-solver: sample matrices, the three Galerkin solvers and the Stein residual.
_rngA004() = Random.MersenneTwister(1234)
_rngB004() = Random.MersenneTwister(5678)
const _T004 = 4
const _L004 = 3
const _N004 = 20
const _κ004 = 0.1
const _p004 = (0.0, 0.1)

# Periodic T × T matrix K0_ij = (2 − 4κ) δ_ij − 2κ (δ_{i,j+1} + δ_{i,j−1}), indices mod T.
function _K0004(T, κ)
    K = zeros(T, T)
    for j in 0:T-1
        K[j+1, j+1] += 2 - 4κ
        K[mod(j + 1, T) + 1, j+1] -= 2κ
        K[mod(j - 1, T) + 1, j+1] -= 2κ
    end
    return K
end

_close004(a, b; tol=1e-12) = isapprox(a, b; rtol=tol, atol=tol)
_exact004(a, b) = isapprox(a, b; rtol=1e-14, atol=1e-14)

@testset "spec 004: level1 solver" begin
    p = _p004
    z = randn(_rngA004(), _T004, _L004, _N004)
    zB = randn(_rngB004(), _T004, _L004, _N004)
    K0 = _K0004(_T004, _κ004)
    G0 = K0 \ Matrix{Float64}(I, _T004, _T004)
    αstar = G0[:, 1]
    e0 = [1.0, 0.0, 0.0, 0.0]

    @testset "AC-1: hand anchors" begin
        a = [-2.0, -1.0, 0.0, 1.0, 2.0]
        za = cat((fill(an, 4, 3) for an in a)...; dims=3)
        m = galerkin_matrices(za, 4, p)
        for name in (:B, :Agrad, :Aibp, :Q)
            @test size(getproperty(m, name)) == (4, 4)
        end
        @test _exact004(m.B, 22.5 .* ones(4, 4))
        @test _exact004(m.Aibp, 27.0 .* ones(4, 4))
        @test _exact004(m.Q, 32.4 .* ones(4, 4))
        @test _exact004(m.Agrad, 3.0 .* Matrix{Float64}(I, 4, 4))

        zc = fill(2.0, 4, 3)
        r1 = stein_residual(zc, 4, p, [1.0, 0.0, 0.0, 0.0])
        r2 = stein_residual(zc, 4, p, ones(4))
        @test _exact004(r1, -1.2)
        @test _exact004(r2, -22.8)
    end

    @testset "AC-2: sample identities of the matrices" begin
        m = galerkin_matrices(z, 4, p)
        ψ = basis(z, 4)
        Lψ = stein_psi(z, 4, p)
        @test size(ψ) == (4, 20)
        @test size(Lψ) == (4, 20)
        for name in (:B, :Agrad, :Aibp, :Q)
            @test size(getproperty(m, name)) == (4, 4)
        end
        @test _close004(m.B, cov(ψ'))
        @test _close004(m.Aibp, -cov(ψ', Lψ'))
        @test _close004(m.Q, cov(Lψ'))
        @test _exact004(m.Agrad, 3.0 .* Matrix{Float64}(I, 4, 4))
        @test _close004(m.Aibp, m.B * K0)
        @test _close004(m.Q, K0 * m.B * K0)
    end

    @testset "AC-3: three solvers" begin
        B = cov(basis(z, 4)')
        αibp = solve_alpha(z, 4, p, :ibp)
        αvm = solve_alpha(z, 4, p, :vm)
        αgrad = solve_alpha(z, 4, p, :grad)
        @test size(αibp) == (4,)
        @test size(αvm) == (4,)
        @test size(αgrad) == (4,)
        @test _close004(αibp, αstar)
        @test _close004(αvm, αstar)
        @test _close004(αgrad, B[:, 1] ./ 3)
        @test norm(αgrad - αstar) > 1e-2
    end

    @testset "AC-4: residuals and cross-fitting" begin
        αibp = solve_alpha(z, 4, p, :ibp)
        αvm = solve_alpha(z, 4, p, :vm)
        αgrad = solve_alpha(z, 4, p, :grad)
        for zz in (zB, z)
            Ribp = stein_residual(zz, 4, p, αibp)
            Rvm = stein_residual(zz, 4, p, αvm)
            Rgrad = stein_residual(zz, 4, p, αgrad)
            @test size(Ribp) == (20,)
            @test size(Rvm) == (20,)
            @test size(Rgrad) == (20,)
            @test isapprox(Ribp, zeros(20); rtol=0, atol=1e-12)
            @test isapprox(Rvm, zeros(20); rtol=0, atol=1e-12)
            expected = vec(-(K0 * αgrad - e0)' * basis(zz, 4))
            @test _close004(Rgrad, expected)
        end
    end

    @testset "AC-5: batch consistency, shapes, independence" begin
        R = stein_residual(z, 4, p, αstar)
        @test size(R) == (20,)
        for n in 1:_N004
            @test R[n] == stein_residual(z[:, :, n], 4, p, αstar)
        end
        z2 = copy(z)
        z2[:, :, 2] = randn(Random.MersenneTwister(99), 4, 3)
        R2 = stein_residual(z2, 4, p, αstar)
        others = [n for n in 1:_N004 if n != 2]
        @test R2[others] == R[others]
    end

    @testset "AC-6: invalid input" begin
        α4 = ones(4)
        for bad in (randn(4, 3), randn(12))
            @test_throws ArgumentError galerkin_matrices(bad, 4, p)
            @test_throws ArgumentError solve_alpha(bad, 4, p, :ibp)
        end
        for bad in (randn(12), randn(4, 3, 5, 2))
            @test_throws ArgumentError stein_residual(bad, 4, p, α4)
        end
        for M in (3, 5)
            @test_throws ArgumentError galerkin_matrices(z, M, p)
            @test_throws ArgumentError solve_alpha(z, M, p, :ibp)
            @test_throws ArgumentError stein_residual(z, M, p, ones(M))
        end
        for bad in (zeros(0, 3, 20), zeros(4, 0, 20), zeros(4, 3, 0))
            @test_throws ArgumentError galerkin_matrices(bad, 4, p)
            @test_throws ArgumentError solve_alpha(bad, 4, p, :ibp)
            @test_throws ArgumentError stein_residual(bad, 4, p, α4)
        end
        zsmall = randn(Random.MersenneTwister(1), 4, 3, 4)
        @test_throws ArgumentError galerkin_matrices(zsmall, 4, p)
        @test_throws ArgumentError solve_alpha(zsmall, 4, p, :ibp)
        @test_throws ArgumentError solve_alpha(z, 4, p, :foo)
        @test_throws DimensionMismatch stein_residual(z, 4, p, ones(3))
    end

    @testset "AC-7: no mutation, element type, dual numbers" begin
        zcopy = copy(z)
        galerkin_matrices(z, 4, p)
        for kind in (:grad, :ibp, :vm)
            solve_alpha(z, 4, p, kind)
        end
        stein_residual(z, 4, p, αstar)
        stein_residual(z[:, :, 1], 4, p, αstar)

        z32 = Float32.(z)
        p32 = (0.0f0, 0.1f0)
        α32 = Float32[1, 0, 0, 0]
        m32 = galerkin_matrices(z32, 4, p32)
        for name in (:B, :Agrad, :Aibp, :Q)
            M32 = getproperty(m32, name)
            @test size(M32) == (4, 4)
            @test eltype(M32) == Float32
        end
        for kind in (:grad, :ibp, :vm)
            a32 = solve_alpha(z32, 4, p32, kind)
            @test size(a32) == (4,)
            @test eltype(a32) == Float32
        end
        R32 = stein_residual(z32, 4, p32, α32)
        @test size(R32) == (20,)
        @test eltype(R32) == Float32

        d = ForwardDiff.derivative(c -> sum(stein_residual(c .* z, 4, p, αstar)), 1.0)
        @test _close004(d, sum(stein_residual(z, 4, p, αstar)))

        @test z == zcopy
    end
end
