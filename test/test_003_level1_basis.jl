using Test
using Random
using ForwardDiff
using LinearAlgebra: tr
using GalerkinDemo

# Spec 003-level1-basis: level-1 basis ψ_j = Π(j), its gradient, Laplacian and Stein operator.
_rng003() = Random.MersenneTwister(1234)
const _T003 = 4
const _L003 = 3
const _κ003 = 0.1
const _p003 = (0.0, 0.1)

# Periodic T × T matrix K0_ij = (2 − 4κ) δ_ij − 2κ (δ_{i,j+1} + δ_{i,j−1}), indices mod T.
function _K0003(T, κ)
    K = zeros(T, T)
    for j in 0:T-1
        K[j+1, j+1] += 2 - 4κ
        K[mod(j + 1, T) + 1, j+1] -= 2κ
        K[mod(j - 1, T) + 1, j+1] -= 2κ
    end
    return K
end

_ψ003(j) = w -> basis(w, _T003)[j+1]

@testset "spec 003: level1 basis" begin
    @testset "AC-1: hand anchors" begin
        z = [Float64(t + 1) for t in 0:_T003-1, x in 0:_L003-1]
        ψ = basis(z, 4)
        @test size(ψ) == (4,)
        @test isapprox(ψ, [3.0, 6.0, 9.0, 12.0]; rtol=1e-14, atol=1e-14)
        @test ψ == [3, 6, 9, 12]

        zc = fill(2.0, 4, 3)
        s = stein_psi(zc, 4, (0.0, 0.1))
        @test size(s) == (4,)
        @test isapprox(s, fill(-7.2, 4); rtol=1e-14, atol=1e-14)
    end

    @testset "AC-2: derivatives vs ForwardDiff" begin
        z = randn(_rng003(), 4, 3)
        G = basis_grad(z, 4)
        lap = basis_laplacian(z, 4)
        @test size(G) == (4, 3, 4)
        @test size(lap) == (4,)

        for j in 0:3
            ψj = _ψ003(j)
            @test isapprox(G[:, :, j+1], ForwardDiff.gradient(ψj, z); rtol=1e-12, atol=1e-12)
            @test isapprox(lap[j+1], tr(ForwardDiff.hessian(ψj, z)); rtol=1e-12, atol=1e-12)
            @test isapprox(lap[j+1], 0.0; rtol=1e-12, atol=1e-12)
        end

        for i in 0:3, j in 0:3
            gram = sum(G[:, :, i+1] .* G[:, :, j+1])
            @test isapprox(gram, i == j ? 3.0 : 0.0; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "AC-3: Stein operator vs ForwardDiff" begin
        z = randn(_rng003(), 4, 3)
        s = stein_psi(z, 4, _p003)
        @test size(s) == (4,)

        ∇S = ForwardDiff.gradient(action_slice(0.0, 0.1), z)
        for j in 0:3
            ψj = _ψ003(j)
            ref = tr(ForwardDiff.hessian(ψj, z)) - sum(∇S .* ForwardDiff.gradient(ψj, z))
            @test isapprox(s[j+1], ref; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "AC-4: closure" begin
        z = randn(_rng003(), 4, 3)
        K0 = _K0003(4, _κ003)
        # sanity check of the helper: symmetric circulant with column sums 2 − 8κ
        @test K0 == K0'
        @test isapprox(vec(sum(K0; dims=1)), fill(2 - 8 * _κ003, 4); rtol=1e-14, atol=1e-14)

        s = stein_psi(z, 4, _p003)
        @test size(s) == (4,)
        @test isapprox(s, -(K0' * basis(z, 4)); rtol=1e-12, atol=1e-12)
    end

    @testset "AC-5: batch consistency, shapes, independence" begin
        rng = _rng003()
        zb = randn(rng, 4, 3, 5)
        # wrap every function as z -> f(z) with M = 4 (and p for stein_psi)
        calls = (
            (z -> basis(z, 4), (4, 5)),
            (z -> basis_grad(z, 4), (4, 3, 4, 5)),
            (z -> basis_laplacian(z, 4), (4, 5)),
            (z -> stein_psi(z, 4, _p003), (4, 5)),
        )
        for (f, sz) in calls
            R = f(zb)
            @test size(R) == sz
            for n in 1:5
                @test selectdim(R, ndims(R), n) == f(zb[:, :, n])
            end
        end

        zb2 = copy(zb)
        zb2[:, :, 2] = randn(rng, 4, 3)
        for (f, _) in calls
            R = f(zb)
            R2 = f(zb2)
            for n in (1, 3, 4, 5)
                @test selectdim(R2, ndims(R2), n) == selectdim(R, ndims(R), n)
            end
        end
    end

    @testset "AC-6: invalid input" begin
        calls = (
            (z, M) -> basis(z, M),
            (z, M) -> basis_grad(z, M),
            (z, M) -> basis_laplacian(z, M),
            (z, M) -> stein_psi(z, M, _p003),
        )
        cases = (
            (randn(12), 4),          # rank 1
            (randn(4, 3, 5, 2), 4),  # rank 4
            (randn(4, 3), 3),        # M < T
            (randn(4, 3), 5),        # M > T
            (zeros(0, 3), 0),        # T = 0
            (zeros(4, 0), 4),        # L = 0
            (zeros(4, 3, 0), 4),     # N = 0
        )
        for f in calls, (z, M) in cases
            @test_throws ArgumentError f(z, M)
        end
    end

    @testset "AC-7: no mutation, element type, dual numbers" begin
        rng = _rng003()
        z = randn(rng, 4, 3)
        zb = randn(rng, 4, 3, 5)
        z0 = copy(z)
        zb0 = copy(zb)
        for w in (z, zb)
            basis(w, 4)
            basis_grad(w, 4)
            basis_laplacian(w, 4)
            stein_psi(w, 4, _p003)
        end
        @test z == z0
        @test zb == zb0

        z32 = Float32.(z)
        p32 = (0f0, 0.1f0)
        @test eltype(basis(z32, 4)) == Float32
        @test eltype(basis_grad(z32, 4)) == Float32
        @test eltype(basis_laplacian(z32, 4)) == Float32
        @test eltype(stein_psi(z32, 4, p32)) == Float32

        lhs = ForwardDiff.derivative(c -> sum(stein_psi(c .* z, 4, _p003)), 1.0)
        rhs = sum(stein_psi(z, 4, _p003))
        @test isapprox(lhs, rhs; rtol=1e-12, atol=1e-12)
    end
end
