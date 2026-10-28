## Rules worker must follow
- Read `CONVENTIONS.md` and the active spec in `specs/` before any code.
- Never edit `test/`, unless I explicitly ask.
- Never loosen a tolerance, change a random seed, or add a skipped test to make a test pass.
- Do not change existing code outside the scope of the spec, and keep existing tests passing.
- Read the conventions document and the spec before writing code. If they conflict with the existing code, say so.
- Reuse existing functions. Do not copy-paste them. If a shared function needs to change, stop and ask.
- Do not commit. I commit after review.

Workflow: restate spec -> list files to touch -> wait for OK -> implement -> run `julia --project -e 'using Pkg; Pkg.test()'` and paste the raw output.