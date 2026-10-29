using Test
using Random
using Statistics: mean, std
using LinearAlgebra
using GalerkinDemo
import ADerrors

# Spec 005-improved-estimator: improved correlator and control variate (free theory, level-1 basis).
const _T005 = 4
const _L005 = 3
const _κ005 = 0.1
const _p005 = (0.0, 0.1)

# Periodic T × T matrix K0_ij = (2 − 4κ) δ_ij − 2κ (δ_{i,j+1} + δ_{i,j−1}), indices mod T.
function _K0005(T, κ)
    K = zeros(T, T)
    for j in 0:T-1
        K[j+1, j+1] += 2 - 4κ
        K[mod(j + 1, T) + 1, j+1] -= 2κ
        K[mod(j - 1, T) + 1, j+1] -= 2κ
    end
    return K
end

# z[:,:,n] ~ N(0, K⁻¹), K the full (TL × TL) matrix built from `grad_slice` (F = Kz is linear).
function gaussian_batch(rng, N)
    T, L = _T005, _L005
    g = grad_slice(0.0, 0.1)
    K = zeros(T * L, T * L)
    for i in 1:T*L
        e = zeros(T * L)
        e[i] = 1.0
        K[:, i] = vec(g(reshape(e, T, L)))
    end
    U = cholesky(Symmetric(K)).U
    z = zeros(T, L, N)
    for n in 1:N
        z[:, :, n] = reshape(U \ randn(rng, T * L), T, L)
    end
    return z
end

_close005(a, b; tol=1e-12) = isapprox(a, b; rtol=tol, atol=tol)
_exact005(a, b) = isapprox(a, b; rtol=1e-14, atol=1e-14)
_stat005(q, q0) = abs(mean(q) - q0) <= 5 * std(q) / sqrt(length(q))

@testset "spec 005: improved estimator" begin
    p = _p005
    T, L = _T005, _L005
    zA = randn(Random.MersenneTwister(1234), 4, 3, 20)
    K0 = _K0005(T, _κ005)
    G0 = K0 \ Matrix{Float64}(I, T, T)
    αstar = G0[:, 1]

    @testset "AC-1: hand anchors" begin
        α = [1.0, 0.0, 0.0, 0.0]
        f = shift_term(zeros(4, 3), 4, α)
        @test size(f) == (4,)
        @test f == [3.0, 0.0, 0.0, 0.0]

        a = [-2.0, -1.0, 0.0, 1.0, 2.0]
        z = zeros(4, 3, 5)
        for n in 1:5
            z[:, :, n] .= a[n]
        end
        R = rolled_residual(z, 4, p, α)
        @test size(R) == (4, 5)
        @test all(_exact005(R[τ, :], -0.6 .* a) for τ in 1:4)

        A = improved_samples(z, 4, p, α)
        @test size(A) == (4, 5)
        @test _exact005(A[1, :], [-4.2, 1.2, 3.0, 1.2, -4.2])
        @test all(_exact005(A[t, :], [-7.2, -1.8, 0.0, -1.8, -7.2]) for t in 2:4)

        CV = control_variate_samples(z, 4, p, α)
        @test size(CV) == (4, 5)
        @test _exact005(CV[1, :], [-40.2, -7.8, 3.0, -7.8, -40.2])
        @test all(_exact005(CV[t, :], [-43.2, -10.8, 0.0, -10.8, -43.2]) for t in 2:4)
    end

    @testset "AC-2: algebraic identities on any sample" begin
        c = correlator_samples(slab_sums(zA))
        for α in (solve_alpha(zA, 4, p, :grad), ones(4))
            A = improved_samples(zA, 4, p, α)
            CV = control_variate_samples(zA, 4, p, α)
            @test size(A) == (4, 20)
            @test size(CV) == (4, 20)
            @test _close005(A, c .+ CV)

            f = shift_term(zA[:, :, 1], 4, α)
            Adir = zeros(4, 20)
            for n in 1:20, t in 0:3
                s = 0.0
                for τ in 0:3
                    R = stein_residual(circshift(zA[:, :, n], (-τ, 0)), 4, p, α)
                    s += R * basis(zA[:, :, n], 4)[mod(τ + t, 4) + 1]
                end
                Adir[t+1, n] = f[t+1] + s / 4
            end
            @test _close005(A, Adir)
        end
    end

    @testset "AC-3: exact α on free-theory samples" begin
        zG = gaussian_batch(Random.MersenneTwister(2024), 50)
        αvm = solve_alpha(zG, 4, p, :vm)
        c = correlator_samples(slab_sums(zG))
        for α in (αstar, αvm)
            R = rolled_residual(zG, 4, p, α)
            A = improved_samples(zG, 4, p, α)
            CV = control_variate_samples(zG, 4, p, α)
            @test size(R) == (4, 50)
            @test size(A) == (4, 50)
            @test size(CV) == (4, 50)
            @test isapprox(R, zeros(4, 50); atol=1e-12)
            @test all(_close005(A[:, n], 3 .* G0[:, 1]) for n in 1:50)
            @test all(_close005(CV[:, n], 3 .* G0[:, 1] .- c[:, n]) for n in 1:50)
        end
    end

    @testset "AC-4: Stein identity and unbiasedness" begin
        zB = gaussian_batch(Random.MersenneTwister(4321), 20000)
        αgrad = solve_alpha(zA, 4, p, :grad)
        ψ = stein_psi(zB, 4, p)
        @test all(_stat005(ψ[t, :], 0.0) for t in 1:4)
        for α in (αgrad, αstar, ones(4))
            CV = control_variate_samples(zB, 4, p, α)
            A = improved_samples(zB, 4, p, α)
            @test size(CV) == (4, 20000)
            @test size(A) == (4, 20000)
            @test all(_stat005(CV[t, :], 0.0) for t in 1:4)
            @test all(_stat005(A[t, :], 3 * G0[t, 1]) for t in 1:4)
        end
    end

    @testset "AC-5: C + CV equals the improved correlator" begin
        z = gaussian_batch(Random.MersenneTwister(99), 2000)
        α = solve_alpha(gaussian_batch(Random.MersenneTwister(7), 40), 4, p, :grad)
        id = "test005-ac5"
        C = correlator(z, id)
        CV = control_variate(z, 4, p, α, id)
        Â = improved_correlator(z, 4, p, α, id)
        @test length(C) == length(CV) == length(Â) == 4
        @test eltype(C) == eltype(CV) == eltype(Â) == ADerrors.uwreal
        S = C .+ CV
        ADerrors.uwerr.(S)
        ADerrors.uwerr.(Â)
        A = improved_samples(z, 4, p, α)
        for t in 1:4
            @test isapprox(ADerrors.value(S[t]), ADerrors.value(Â[t]); rtol=1e-10)
            @test isapprox(ADerrors.err(S[t]), ADerrors.err(Â[t]); rtol=1e-8, atol=1e-12)
            @test _close005(ADerrors.value(Â[t]), mean(A[t, :]))
        end
    end

    @testset "AC-6: exact α gives zero error" begin
        z = gaussian_batch(Random.MersenneTwister(99), 2000)
        id = "test005-ac6"
        Â = improved_correlator(z, 4, p, αstar, id)
        C = correlator(z, id)
        CV = control_variate(z, 4, p, αstar, id)
        S = C .+ CV
        ADerrors.uwerr.(Â)
        ADerrors.uwerr.(C)
        ADerrors.uwerr.(S)
        for t in 1:4
            @test isapprox(ADerrors.value(Â[t]), 3 * G0[t, 1]; rtol=1e-10)
            @test ADerrors.err(Â[t]) <= 1e-9
            @test ADerrors.err(C[t]) > 1e-3
            @test isapprox(ADerrors.value(S[t]), 3 * G0[t, 1]; rtol=1e-10)
        end
    end

    @testset "AC-7: invalid input" begin
        α = ones(4)
        z = randn(Random.MersenneTwister(1), 4, 3, 20)
        batch_fns = (rolled_residual, improved_samples, control_variate_samples)
        uw_fns = (improved_correlator, control_variate)
        call(f, z, M, α) = f in uw_fns ? f(z, M, p, α, "test005-ac7") :
                           f === shift_term ? f(z, M, α) : f(z, M, p, α)

        for f in batch_fns
            @test_throws ArgumentError call(f, randn(4, 3), 4, α)
            @test_throws ArgumentError call(f, randn(12), 4, α)
        end
        @test_throws ArgumentError shift_term(randn(4, 3, 5), 4, α)
        @test_throws ArgumentError shift_term(randn(12), 4, α)

        for f in (batch_fns..., uw_fns...)
            @test_throws ArgumentError call(f, z, 3, α)
            @test_throws ArgumentError call(f, z, 5, α)
            @test_throws ArgumentError call(f, zeros(0, 3, 20), 4, α)
            @test_throws ArgumentError call(f, zeros(4, 0, 20), 4, α)
            @test_throws ArgumentError call(f, zeros(4, 3, 0), 4, α)
            @test_throws DimensionMismatch call(f, z, 4, ones(3))
        end
        @test_throws ArgumentError shift_term(zeros(0, 3), 4, α)
        @test_throws ArgumentError shift_term(zeros(4, 0), 4, α)
        @test_throws ArgumentError shift_term(zeros(4, 3), 3, α)
        @test_throws ArgumentError shift_term(zeros(4, 3), 5, α)
        @test_throws DimensionMismatch shift_term(zeros(4, 3), 4, ones(3))

        for f in uw_fns
            @test_throws ArgumentError call(f, randn(4, 3, 2), 4, α)
            zn = copy(z)
            zn[2, 2, 3] = NaN
            @test_throws ArgumentError call(f, zn, 4, α)
        end
    end

    @testset "AC-8: no mutation, element type" begin
        z0 = copy(zA)
        α0 = ones(4)
        αc = copy(α0)
        shift_term(zA[:, :, 1], 4, α0)
        rolled_residual(zA, 4, p, α0)
        improved_samples(zA, 4, p, α0)
        control_variate_samples(zA, 4, p, α0)
        improved_correlator(zA, 4, p, α0, "test005-ac8-a")
        control_variate(zA, 4, p, α0, "test005-ac8-b")
        @test zA == z0
        @test α0 == αc

        z32 = Float32.(zA)
        p32 = (0.0f0, 0.1f0)
        α32 = Float32[1, 0, 0, 0]
        @test eltype(shift_term(z32[:, :, 1], 4, α32)) == Float32
        @test eltype(rolled_residual(z32, 4, p32, α32)) == Float32
        @test eltype(improved_samples(z32, 4, p32, α32)) == Float32
        @test eltype(control_variate_samples(z32, 4, p32, α32)) == Float32
    end
end
