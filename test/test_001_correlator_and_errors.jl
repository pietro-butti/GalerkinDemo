using Test
using Random
using Statistics
using GalerkinDemo
import ADerrors
import Plots

# "exact" in the spec: rtol=1e-14, atol=1e-14
exact(a, b) = isapprox(a, b; rtol=1e-14, atol=1e-14)

@testset "spec 001: correlator and errors" begin
    @testset "AC-1: slab sums" begin
        z = zeros(2, 3, 3)
        z[:, :, 1] = [1 2 3; 4 5 6]
        z[:, :, 2] = [-1 0 1; 2 2 2]
        z[:, :, 3] = [0 0 0; 1 1 1]
        P = GalerkinDemo.slab_sums(z)
        @test size(P) == (2, 3)
        @test all(exact.(P, [6 0 0; 15 6 3]))
    end

    @testset "AC-2: rolled correlator by hand" begin
        Π = [1 1; 2 0; 3 0; 4 0]
        c = GalerkinDemo.correlator_samples(Π)
        @test size(c) == (4, 2)
        @test all(exact.(c, [7.5 0.25; 6 0; 5.5 0; 6 0]))

        c1 = GalerkinDemo.correlator_samples([3.0;;])
        @test size(c1) == (1, 1)
        @test exact(c1[1, 1], 9.0)
    end

    @testset "AC-3: periodic symmetry" begin
        z = randn(Random.MersenneTwister(1234), 8, 3, 10)
        c = GalerkinDemo.correlator_samples(GalerkinDemo.slab_sums(z))
        for t in 1:7
            @test isapprox(c[t+1, :], c[8-t+1, :]; rtol=1e-12)
        end
    end

    @testset "AC-4: ensemble mean" begin
        Π = [1 1 1 1; 2 0 0 0; 3 0 0 0; 4 0 0 0]
        z = zeros(4, 1, 4)
        z[:, 1, :] = Π
        C = GalerkinDemo.correlator(z, "test001-ac4")
        @test length(C) == 4
        @test all(exact.(ADerrors.value.(C), [2.0625, 1.5, 1.375, 1.5]))
    end

    @testset "AC-5: shared id correlates the t's" begin
        z = randn(Random.MersenneTwister(1234), 8, 3, 20)
        C = GalerkinDemo.correlator(z, "test001-ac5")
        D = C[2] - C[8]
        ADerrors.uwerr(D)
        @test isapprox(ADerrors.value(D), 0.0; atol=1e-12)
        @test isapprox(ADerrors.err(D), 0.0; atol=1e-12)
    end

    @testset "AC-6: constant data" begin
        z = ones(8, 4, 50)
        C = GalerkinDemo.correlator(z, "test001-ac6")
        stats = GalerkinDemo.correlator_stats(C, "test001-ac6")
        @test all(exact.(stats.value, 16.0))
        @test all(isapprox.(stats.err, 0.0; atol=1e-14))
    end

    @testset "AC-7: stats structure" begin
        z = randn(Random.MersenneTwister(1234), 8, 3, 2000)
        C = GalerkinDemo.correlator(z, "test001-ac7")
        stats = GalerkinDemo.correlator_stats(C, "test001-ac7")
        @test stats.t == 0:7
        @test length(stats.value) == 8
        @test length(stats.err) == 8
        @test length(stats.taui) == 8
        @test length(stats.window) == 8
        @test stats.value == ADerrors.value.(C)
        @test stats.err == ADerrors.err.(C)
        @test all(stats.err .>= 0)
    end

    @testset "AC-8: iid error scale" begin
        z = randn(Random.MersenneTwister(1234), 8, 3, 2000)
        N = size(z, 3)
        c = GalerkinDemo.correlator_samples(GalerkinDemo.slab_sums(z))
        C = GalerkinDemo.correlator(z, "test001-ac8")
        stats = GalerkinDemo.correlator_stats(C, "test001-ac8")
        for t in 0:7
            @test isapprox(stats.err[t+1], std(c[t+1, :]) / sqrt(N); rtol=0.2)
        end
    end

    @testset "AC-9: no mutation, no I/O" begin
        z = randn(Random.MersenneTwister(1234), 8, 3, 10)
        z0 = copy(z)
        GalerkinDemo.slab_sums(z)
        @test z == z0

        Π = GalerkinDemo.slab_sums(z)
        Π0 = copy(Π)
        GalerkinDemo.correlator_samples(Π)
        @test Π == Π0

        GalerkinDemo.correlator(z, "test001-ac9")
        @test z == z0
    end

    @testset "AC-10: invalid input" begin
        @test_throws ArgumentError GalerkinDemo.correlator(zeros(0, 3, 3), "test001-ac10-a")
        @test_throws ArgumentError GalerkinDemo.correlator(zeros(4, 0, 3), "test001-ac10-b")
        @test_throws ArgumentError GalerkinDemo.correlator(zeros(4, 3, 1), "test001-ac10-c")
        @test_throws ArgumentError GalerkinDemo.correlator(zeros(4, 3, 2), "test001-ac10-g")

        zn = ones(4, 3, 3); zn[1, 1, 1] = NaN
        @test_throws ArgumentError GalerkinDemo.correlator(zn, "test001-ac10-d")
        zi = ones(4, 3, 3); zi[2, 2, 2] = Inf
        @test_throws ArgumentError GalerkinDemo.correlator(zi, "test001-ac10-e")

        @test_throws ArgumentError GalerkinDemo.correlator_stats(ADerrors.uwreal[], "test001-ac10-f")

        @test_throws ArgumentError GalerkinDemo.plottable(ADerrors.uwreal[])
    end

    @testset "AC-11: plottable" begin
        z = randn(Random.MersenneTwister(1234), 8, 3, 2000)
        C = GalerkinDemo.correlator(z, "test001-ac11")

        # (a) tol = -Inf keeps everything
        s = GalerkinDemo.plottable(C; tol=-Inf)
        @test s.t == 0:7
        @test s.value == ADerrors.value.(C)
        @test s.err == ADerrors.err.(C)

        # (b) tol = Inf keeps nothing
        s = GalerkinDemo.plottable(C; tol=Inf)
        @test isempty(s.t)
        @test isempty(s.value)
        @test isempty(s.err)

        # (c) default tol: kept points pass, dropped points fail
        s = GalerkinDemo.plottable(C)
        @test all(s.value .- s.err .> 1e-10)
        dropped = setdiff(0:7, s.t)
        v = ADerrors.value.(C)
        e = ADerrors.err.(C)
        for t in dropped
            @test v[t+1] - e[t+1] <= 1e-10
        end

        # hand-checkable: constant data keeps all points
        Cc = GalerkinDemo.correlator(ones(8, 4, 50), "test001-ac11-const")
        sc = GalerkinDemo.plottable(Cc)
        @test sc.t == 0:7
        @test all(exact.(sc.value, 16.0))
        @test all(exact.(sc.err, 0.0))
    end
end
