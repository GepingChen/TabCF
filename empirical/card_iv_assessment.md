# Card Benchmark: Background, Variables, and Identification Issues

This note documents the `wooldridge::card` benchmark that is now available in the
`empirical/` pipeline.

## 1. Background

The `card` dataset is the standard teaching version of David Card's classic
"college proximity" design for the return to schooling. The economic question is
whether additional education raises wages. The instrumental-variables idea is that
living near a four-year college lowers the cost of attending college and therefore
shifts schooling decisions. If proximity is otherwise unrelated to latent wage
determinants, it can be used as an instrument for education.

The Wooldridge version used here contains `3010` male observations with wages
observed in `1976`. In empirical practice, Card-style specifications are usually
estimated with a rich set of controls. Our benchmark in this repo is deliberately
much simpler: we keep only the instrument, the treatment, and the outcome so that
we can test how the current scalar-IV TabCF pipeline behaves on a well-known real
dataset.

## 2. Variables Used in This Repo

We map the raw dataset to the canonical scalar-IV notation as follows.

| Canonical role | Raw variable | Meaning | Type in this benchmark |
| --- | --- | --- | --- |
| `Z` | `nearc4` | Indicator for whether the person lived near a four-year college in 1966 | Binary instrument |
| `X` | `educ` | Years of completed schooling by 1976 | Ordered, discrete treatment |
| `Y` | `lwage` | Log wage in 1976 | Continuous outcome |

The observed treatment support in this benchmark is:

`educ ∈ {1, 2, ..., 18}`.

This matters because the treatment is not continuously distributed. For that
reason, the pipeline evaluates all methods only on the observed support points,
rather than on an artificial continuous grid.

## 3. Other Variables Present in the Raw File

The raw `card` file contains many additional variables that are often used in
applied work:

- family background: `fatheduc`, `motheduc`, `momdad14`, `sinmom14`, `step14`
- demographics: `black`, `age`, `married`
- geography and urban status: `smsa`, `smsa66`, `south`, `south66`, `reg661` to `reg669`
- labor-market controls: `exper`, `expersq`
- ability / test proxies: `IQ`, `KWW`
- alternative proximity measure: `nearc2`

These variables are relevant because the credibility of the Card instrument is
usually discussed conditional on such controls. Our stripped-down benchmark does
not include them, so the identification assumptions are much stronger here than in
the original empirical application.

## 4. How the Repo Uses the Data

The benchmark can be run with:

```bash
DATASET=card bash empirical/run_empirical_mean.sh
```

The pipeline downloads the raw CSV to:

`empirical/downloads/card_wooldridge.csv`

and writes outputs to:

`empirical/outputs_card/`

In the current implementation:

- Stage 1 estimates the control function through `F_{X|Z}(x | z)`
- Stage 2 fits the structural regression using `(X, V_hat)`
- prediction is reported only at the observed education levels `1, ..., 18`

## 5. Empirical Snapshot from the Current Run

On the current machine, the full run completed successfully. The benchmark is
therefore computationally feasible for the present codebase.

Key summary numbers:

- first-stage shift in schooling:
  - `E[educ | nearc4 = 0] = 12.698`
  - `E[educ | nearc4 = 1] = 13.527`
  - difference: `0.829`
- reduced-form shift in wages:
  - `E[lwage | nearc4 = 0] = 6.155`
  - `E[lwage | nearc4 = 1] = 6.311`
- linear benchmarks:
  - OLS slope: `0.0521`
  - 2SLS / Wald slope: `0.1881`
- TabCF prediction range on `educ = 1, ..., 18`:
  - minimum: `5.4646`
  - maximum: `6.5942`

Relative to the linear 2SLS line, the fitted TabCF curve is higher at low
schooling levels, similar around `educ = 12` to `13`, and lower at high schooling
levels. So the model is learning a nonlinear shape rather than mechanically
replicating the Wald slope.

## 6. Identification Issues Relative to the TabCF Conditions

Below we interpret the Card benchmark through the identification conditions used
by the TabCF control-function framework.

### CF1. Exclusion / triangular structure

The intended economic story is:

- `nearc4` affects wages only through schooling,
- proximity changes the cost of attending college,
- once schooling is fixed, college proximity has no direct wage effect.

This is plausible as a motivating instrument, but it is strong in our simplified
benchmark. Living near a four-year college may also proxy for broader local
conditions such as urban access, school quality, labor-market opportunities,
regional development, or family background. In the original Card-style empirical
strategy, these concerns are usually addressed by conditioning on geography and
other controls. Because our benchmark omits those controls, unconditional
exclusion is much harder to defend.

### CF2. Instrument exogeneity

TabCF assumes the instrument is independent of the latent disturbances driving the
first-stage and outcome equations. In notation, this is the analogue of
`Z ⟂ (η, ε)`.

For the Card benchmark, this is again much more credible after conditioning on
observables than in the raw three-variable design. Proximity to college can be
correlated with parental education, urban residence, region, race, or latent
ability. The raw file contains many of these covariates, which is itself a signal
that unconditional exogeneity is not the natural maintained assumption. Therefore,
CF2 is doubtful in the present stripped-down benchmark.

### CF3. Monotone first stage and continuous rank condition

This is the most direct mismatch with the standard TabCF theory.

The TabCF identification argument uses a scalar latent first-stage disturbance
`η`, a monotone first-stage map `X = h(Z, η)`, and a continuously varying
conditional distribution `F_{X|Z}(· | z)` that is strictly increasing on the
support of `X | Z = z`. Under those conditions, the control variable
`V = F_{X|Z}(X | Z)` behaves like a continuous rank.

In `card`, however:

- `Z = nearc4` is binary,
- `X = educ` is discrete and integer-valued,
- `F_{X|Z}(· | z)` is a step function rather than a continuous strictly increasing CDF.

As a result, the continuous-rank interpretation of `V` is no longer available in
the same way. The current TabCF code can still compute a fitted control function,
but the clean theoretical justification from the continuous triangular model does
not carry over directly.

### CF4. Common support / overlap in `(X, V)`

The support condition used by TabCF requires enough variation in the control rank
at each treatment value. Intuitively, for every target `x`, the model should see a
rich enough range of latent ranks `v` to identify the structural function
`m(x, v)`.

This is problematic here for two reasons:

- the instrument has only two support points, so it generates only two conditional
  schooling distributions,
- the treatment has many support points, so we are trying to learn an 18-point
  structural response from very limited first-stage variation.

This is enough to define a Wald contrast and to run linear 2SLS under strong
restrictions, but it is not enough to nonparametrically recover a full
`E[Y | do(X = x)]` curve over all schooling levels.

## 7. What Is and Is Not Identified Here

The main distinction is the following.

- A linear IV estimand such as the Wald ratio or a 2SLS coefficient can still be
  interpreted under its own maintained assumptions.
- A fully nonparametric schooling dose-response curve over `educ = 1, ..., 18`
  is not generally identified from this binary-instrument, multivalued-treatment
  design.

Therefore, when TabCF outputs a smooth nonlinear curve on the Card benchmark, that
curve should be interpreted as a model-based regularized fit, not as a fully
identified causal response function justified by the standard continuous
control-function theory.

## 8. Bottom Line

Implementation-wise, yes: the current TabCF pipeline can be run on the Card
benchmark once the treatment grid is treated as discrete.

Identification-wise, no in the strong sense: the Card benchmark is not a clean
match to the continuous-rank TabCF assumptions. It is useful as:

- a stress test for the code,
- a nonlinear benchmark against `OLS`, `2SLS`, and `DIV`,
- an illustration of where the theory begins to stretch.

It should not be presented as a setting in which the full TabCF dose-response
curve is cleanly identified without additional structure or stronger assumptions.
