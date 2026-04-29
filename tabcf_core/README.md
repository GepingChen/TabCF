# tabcf_core

Shared scalar-IV TabCF implementation used by the paper-facing benchmark
blocks.

The public release keeps the Stage 1 control-function code, Stage 2 outcome
model code, foundation-model backend helpers, local-context helpers, and
integration utilities here so that `interv_mean`, `interv_qtl`, `multivar`, and
`empirical` all depend on one shared implementation.

The public release exposes this as a first-class core module because it is
reused outside the mean benchmark.
