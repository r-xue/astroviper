# GitHub Copilot Custom Instructions for AstroVIPER

You are an expert Python developer assisting with the **AstroVIPER**
(Astro Visibility and Image Parallel Execution Reduction) codebase — a
radio-astronomy imaging library built on `xarray`, `numpy`, `numba`, and
optionally `finufft` / `cufinufft`.

## Project Facts

| Property | Value |
|---|---|
| **Package name** | `astroviper` |
| **Source layout** | `src/astroviper/` |
| **Python support** | `>=3.11, <3.14` (3.11 · 3.12 · 3.13) |
| **Build system** | scikit-build-core + pybind11 + CMake (C++ extensions) |
| **Core deps** | `graphviper`, `numpy`, `xradio[zarr]`, `numba` |
| **Optional deps** | `finufft`, `cufinufft`, `cupy`, `python_casacore`, `matplotlib` |
| **Test framework** | `pytest` (with `pytest-cov`, `pytest-html`) |
| **CI** | GitHub Actions — Linux, macOS, notebook tests, Codecov |

## 1. Code Style & Generation Rules

**Target Python 3.11+. Use features available from 3.11 onwards.**

* **Type Hints (PEP 604 / PEP 585):**
    * Add type hints to all function arguments and return types.
    * Use built-in generics: `list[int]`, `dict[str, float]`, `tuple[int, ...]`.
    * Use union syntax: `int | None` instead of `Optional[int]`.
    * Never import from `typing` for `List`, `Dict`, `Tuple`, `Optional`, `Union`.
* **Formatting:**
    * 4-space indentation.
    * Prefer single quotes (`'`) over double quotes (`"`).
    * Max line width: 100 characters.
* **Logging:** Use lazy formatting (`logger.info('Msg: %s', var)`) — never
  f-strings inside logging calls.
* **Imports:**
    * Group: stdlib → third-party → `astroviper` internals, separated by
      blank lines.
    * Prefer explicit imports over star imports.
* **NumPy / Numba:** Follow existing patterns — Numba-JIT functions use
  `@njit` with explicit signatures where present. Do not add Numba to code
  paths that are not already JIT-compiled without discussion.
* **xarray / xradio:** MSv4 datasets use dimension names `time`,
  `baseline_id`, `frequency`, `polarization` and data variables
  `VISIBILITY`, `UVW`, `WEIGHT`, `FLAG`, `VISIBILITY_MODEL`.
* **Optional GPU deps:** Guard `cupy` / `cufinufft` imports behind
  try/except with a user-friendly `ImportError` message.
* **Avoid major structural changes** unless explicitly requested.

## 2. Documentation Rules (Docstrings)

* **Format:** Numpy-style docstrings (PEP 257 compatible).
* **No redundant types:** Do **not** repeat type information in `Args:` or
  `Returns:` descriptions — rely on the function signature annotations.
* **Language:** Preserve existing notes and warnings close to their original
  meaning; correct grammar and awkward phrasing.
* **Module-level docstrings:** Include a one-line summary of the module's
  purpose.

## 3. Code Review Guidelines

When reviewing code, prioritise:

1. **Python version:** Flag any syntax or API not available in 3.11+, or
   use of deprecated typing imports.
2. **Logging safety:** Ensure lazy formatting in all logging calls.
3. **Readability:** Identify lines > 100 chars or duplicated logic that
   should be refactored.
4. **Correctness:** Check array shapes, dtype mismatches, and missing
   `.copy()` on mutable slices.
5. **Optional deps:** Verify that `finufft`, `cufinufft`, `cupy`,
   `python_casacore`, and `matplotlib` are not imported at module level
   without a guard.
6. **Tone:** Be constructive and concise.

## 4. Testing

* Tests live under `tests/` and use `pytest`.
* When generating unit tests, use `pytest` fixtures and parametrise over
  relevant dimensions (e.g., backend name, polarisation count).
* Use `pytest.importorskip('finufft')` (or `cufinufft`, `cupy`) to skip
  tests that require optional dependencies.

## 5. Git & Commit Messages

* Use **Conventional Commits** format: `feat:`, `fix:`, `refactor:`,
  `docs:`, `test:`, `ci:`, `chore:`.
* Keep the subject line ≤ 72 characters; use the body for detail.
* Reference GitHub issues where applicable (`Fixes #123`).