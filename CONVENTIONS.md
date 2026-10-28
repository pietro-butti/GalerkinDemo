# Conventions
- Language: Julia >= 1.10. Package name: GalerkinDemo.
  (rows = configurations, columns = basis functions).
- Covariance: sample covariance with 1/(N-1) normalisation (Statistics.cov).
- Functions are pure (no global state, no I/O) unless the spec says otherwise.
- Linear solves: use `\`, never `inv`.
- Tests: Test.jl, one @testset per acceptance criterion, named "AC-<n>: <short title>".
- Random numbers: `Random.MersenneTwister(<seed>)`, seed fixed in the test.
- Tolerances: always stated in the spec. Tests use `isapprox(a, b; rtol=..., atol=...)`.
- Errors: invalid input throws (DimensionMismatch / ArgumentError), never returns NaN silently.
- Docstrings: every exported function states its formula.