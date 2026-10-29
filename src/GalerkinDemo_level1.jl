# Level-1 Galerkin basis for the free theory (spec 003-level1-basis).
# Included from GalerkinDemo.jl; uses `grad_slice` from spec 002.

export basis, basis_grad, basis_laplacian, stein_psi
export galerkin_matrices, solve_alpha, stein_residual

# Throws unless `z` is a (T, L) array with T, L >= 1 and `M == T`.
function _check_basis_args(z::AbstractMatrix{<:Real}, M)
    T, L = size(z)
    (T >= 1 && L >= 1) ||
        throw(ArgumentError("z must have at least one time slab and one site, got size $(size(z))"))
    M == T ||
        throw(ArgumentError("level-1 basis requires M == T, got M = $M, T = $T"))
    return nothing
end

# Throws unless `z` is a (T, L, N) array with T, L, N >= 1 and `M == T`.
function _check_basis_args(z::AbstractArray{<:Real,3}, M)
    size(z, 3) >= 1 ||
        throw(ArgumentError("z must have at least one configuration, got size $(size(z))"))
    _check_basis_args(view(z, :, :, 1), M)
    return nothing
end

_rank_error(z) = throw(ArgumentError("z must have rank 2 (T, L) or 3 (T, L, N), got rank $(ndims(z))"))

"""
    basis(z, M) -> Array

Level-1 basis values, `ψ_j = Π(j) = Σ_x z(j,x)`, `j = 0..M-1` (entry `j+1`), with `M == T`.
`z` of size `(T, L)` gives a vector of size `(M,)`; `z` of size `(T, L, N)` gives size
`(M, N)`, column `n` being the result for `z[:,:,n]`.
"""
function basis(z::AbstractMatrix{<:Real}, M)
    _check_basis_args(z, M)
    return [sum(selectdim(z, 1, j)) for j in 1:M]
end
basis(z::AbstractArray{<:Real,3}, M) =
    (_check_basis_args(z, M); stack(map(s -> basis(s, M), eachslice(z; dims=3))))
basis(z::AbstractArray, M) = _rank_error(z)

"""
    basis_grad(z, M) -> Array

Gradient of the level-1 basis, `∂ψ_j/∂z(t,x) = δ_tj`. `z` of size `(T, L)` gives size
`(T, L, M)` with entry `[t+1, x+1, j+1]`; `z` of size `(T, L, N)` gives `(T, L, M, N)`.
The values do not depend on `z`; the element type is `eltype(z)`.
"""
function basis_grad(z::AbstractMatrix{<:Real}, M)
    _check_basis_args(z, M)
    T, L = size(z)
    G = zeros(eltype(z), T, L, M)
    for j in 1:M
        G[j, :, j] .= one(eltype(z))
    end
    return G
end
basis_grad(z::AbstractArray{<:Real,3}, M) =
    (_check_basis_args(z, M); stack(map(s -> basis_grad(s, M), eachslice(z; dims=3))))
basis_grad(z::AbstractArray, M) = _rank_error(z)

"""
    basis_laplacian(z, M) -> Array

Laplacian of the level-1 basis, `∇²ψ_j = Σ_{t,x} ∂²ψ_j/∂z(t,x)² = 0` (ψ_j is linear).
Size `(M,)` for `z` of size `(T, L)`, `(M, N)` for `(T, L, N)`; element type `eltype(z)`.
"""
function basis_laplacian(z::AbstractMatrix{<:Real}, M)
    _check_basis_args(z, M)
    return zeros(eltype(z), M)
end
basis_laplacian(z::AbstractArray{<:Real,3}, M) =
    (_check_basis_args(z, M); stack(map(s -> basis_laplacian(s, M), eachslice(z; dims=3))))
basis_laplacian(z::AbstractArray, M) = _rank_error(z)

"""
    stein_psi(z, M, p) -> Array

Stein operator on the level-1 basis, `ℒψ_j = ∇²ψ_j − ∇S·∇ψ_j = − Σ_x F(j,x)`, with
`F = grad_slice(p...)(z)` and `p = (λ, κ)`. Size `(M,)` for `z` of size `(T, L)`,
`(M, N)` for `(T, L, N)`. Only `λ = 0` is specified and tested.
"""
function stein_psi(z::AbstractMatrix{<:Real}, M, p)
    _check_basis_args(z, M)
    F = grad_slice(p...)(z)
    return [-sum(selectdim(F, 1, j)) for j in 1:M]
end
stein_psi(z::AbstractArray{<:Real,3}, M, p) =
    (_check_basis_args(z, M); stack(map(s -> stein_psi(s, M, p), eachslice(z; dims=3))))
stein_psi(z::AbstractArray, M, p) = _rank_error(z)

# ---- Galerkin matrices, α solvers and residual (spec 004-level1-solver) ----

# Throws unless `z` is a (T, L, N) batch with T, L >= 1, M == T and N > T.
function _check_solver_args(z::AbstractArray{<:Real,3}, M)
    _check_basis_args(z, M)
    size(z, 3) > size(z, 1) ||
        throw(ArgumentError("need N > T for a non-singular B, got N = $(size(z, 3)), T = $(size(z, 1))"))
    return nothing
end

_batch_rank_error(z) = throw(ArgumentError("z must have rank 3 (T, L, N), got rank $(ndims(z))"))

"""
    galerkin_matrices(z, M, p) -> NamedTuple (B, Agrad, Aibp, Q)

Sample matrices of the level-1 basis for a `(T, L, N)` batch `z`, `p = (λ, κ)`, `M == T`,
`N > T`; each is `(M, M)`, indices `i, j = 0..M-1`:

    B_ij     = cov(ψ_i, ψ_j)
    Agrad_ij = (1/N) Σ_n ∇ψ_i·∇ψ_j (z_n)
    Aibp_ij  = − cov(ψ_i, ℒψ_j)
    Q_ij     = cov(ℒψ_i, ℒψ_j)

with the sample covariance `cov(f, g) = (1/(N−1)) Σ_n (f_n − f̄)(g_n − ḡ)`. `Aibp` is not
symmetrised.
"""
function galerkin_matrices(z::AbstractArray{<:Real,3}, M, p)
    _check_solver_args(z, M)
    T, L, N = size(z)
    ψ = basis(z, M)
    Lψ = stein_psi(z, M, p)
    G = reshape(basis_grad(z, M), T * L, M, N)
    Agrad = sum(n -> G[:, :, n]' * G[:, :, n], 1:N) / N
    return (B = cov(ψ'), Agrad = Agrad, Aibp = -cov(ψ', Lψ'), Q = cov(Lψ'))
end
galerkin_matrices(z::AbstractArray, M, p) = _batch_rank_error(z)

"""
    solve_alpha(z, M, p, kind) -> Vector

Galerkin coefficients `α` (size `(M,)`) for the source `s = e₀` (`s_j = δ_j0`), from the
matrices of [`galerkin_matrices`](@ref); `kind` selects the system, solved with `\\`:

    :grad   Agrad α = B s
    :ibp    Aibp  α = B s
    :vm     Q α = Aibpᵀ s
"""
function solve_alpha(z::AbstractArray{<:Real,3}, M, p, kind::Symbol)
    kind in (:grad, :ibp, :vm) ||
        throw(ArgumentError("kind must be :grad, :ibp or :vm, got :$kind"))
    (; B, Agrad, Aibp, Q) = galerkin_matrices(z, M, p)
    s = zeros(eltype(B), M)
    s[1] = one(eltype(B))
    kind === :grad && return Agrad \ (B * s)
    kind === :ibp && return Aibp \ (B * s)
    return Q \ (Aibp' * s)
end
solve_alpha(z::AbstractArray, M, p, kind::Symbol) = _batch_rank_error(z)

"""
    stein_residual(z, M, p, α) -> Real or Vector

Pointwise residual of the coefficients `α` (length `M`):

    R(z; α) = ψ₀(z) + Σ_j α_j ℒψ_j(z)

A `(T, L)` configuration gives a scalar, a `(T, L, N)` batch a vector of size `(N,)`.
"""
function stein_residual(z::AbstractMatrix{<:Real}, M, p, α)
    _check_basis_args(z, M)
    length(α) == M ||
        throw(DimensionMismatch("α has length $(length(α)), expected M = $M"))
    return basis(z, M)[1] + sum(α .* stein_psi(z, M, p))
end
function stein_residual(z::AbstractArray{<:Real,3}, M, p, α)
    _check_basis_args(z, M)
    length(α) == M ||
        throw(DimensionMismatch("α has length $(length(α)), expected M = $M"))
    return map(s -> stein_residual(s, M, p, α), eachslice(z; dims=3))
end
stein_residual(z::AbstractArray, M, p, α) = _rank_error(z)
