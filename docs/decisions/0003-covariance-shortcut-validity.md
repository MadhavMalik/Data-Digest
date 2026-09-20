# 0003 — Complete-case covariance, and refusing the shortcut when it breaks

## Problem

Testing a linear combination normally costs an N-row pass. For `z = aᵀX`, `w = bᵀX`:

```
Cov(z,w) = aᵀ Σ b      corr(z,w) = aᵀΣb / √(aᵀΣa · bᵀΣb)
```

So Σ computed once turns every subsequent linear combination into k×k arithmetic. With
N = 3.5 M and k = 19 that is ~10⁴× per candidate. The question is not whether to do it — it
is what to do about **missing values**, which the algebra is silent about.

## The trap

The identity holds only when z and w are evaluated on the rows Σ was estimated from.

The tempting implementation is **pairwise-complete** Σ: estimate each entry `Σᵢⱼ` from
whatever rows columns i and j happen to share. This maximizes data use per entry and is what
`pandas.DataFrame.cov()` does by default. It is also **not the covariance matrix of any
single dataset**: it need not be positive semi-definite, `aᵀΣa` can come out negative, and
the resulting "correlation" matches no computation anyone could reproduce.

On the real TLC file this is not hypothetical — `passenger_count`, `RatecodeID` and
`congestion_surcharge` are each ~29% null, on overlapping but non-identical rows.

## Candidate designs

1. **Pairwise-complete Σ** — maximum data, silently invalid algebra.
2. **Complete-case Σ** — listwise deletion, algebra exactly valid on the retained subset.
3. **Imputation** — fill missing values, then complete-case.
4. **Complete-case + an explicit validity gate + fallback.**

## Chosen

**(4).** Σ is estimated on complete cases only. The model records `n_complete`, `n_total`
and `complete_fraction`. `is_valid()` **refuses** the shortcut below 50% complete or under 30
complete cases, and `invalid_reason()` says which. Callers fall back to direct vectorized
computation on the projected columns.

## Why

- A **slower correct answer beats a faster meaningless one.** This is the whole point of
  having a deterministic truth layer.
- The gate is **explicit and reported**, so the demo panel shows whether the shortcut was
  actually used rather than implying it always is.
- Complete-case is honest about what it computed: statistics over a stated subset, with the
  retained fraction attached to every result as a warning.

## Rejected

- **(1) Pairwise-complete** — the failure is silent, which makes it the most dangerous option.
- **(3) Imputation** — invents data. In a system whose purpose is separating signal from
  noise, imputing values and then reporting correlations over them would undermine the
  premise. Also requires choosing an imputation model, which is itself an unvalidated
  assumption.

## Implications

- **Runtime:** ~10⁴× per linear-combination candidate when valid; on the real file, valid at
  71.7% complete over 19 variables.
- **Accuracy:** agreement with direct computation measured at **2.8 × 10⁻¹⁷**.
- **Memory:** one k×k float64 matrix — trivial.
- **Honesty:** every shortcut result carries a warning naming the complete-case fraction.

## Tests

`tests/unit/test_statistics.py::TestCovarianceShortcut`

- `test_shortcut_matches_direct_computation` — agreement to 1e-10 absolute
- `test_shortcut_across_many_weightings` — three different weight vectors
- `test_shortcut_refuses_itself_when_missingness_breaks_the_algebra` — constructs 70%/50%
  column-specific missingness and asserts the model refuses with a reason
- `test_complete_case_covariance_is_internally_consistent` — Σ diagonal equals the variances
