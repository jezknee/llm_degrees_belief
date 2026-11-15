# Report card: Qwen/Qwen3-32B

Run `qwen3-32b`, 2026-11-14. **NOT CERTIFIED: see the failed checks**

## Gate (smoke test and Stage 1)

- matching: dominance at the top of the grid 80%, curves crossing 0.5 or confident 100% (0 confident beliefs), mean order gap 0.45
- ramsey: dominance at the top of the grid 100%, curves crossing 0.5 or confident 100% (0 confident beliefs), mean order gap 0.25
- Stage 1 dominance: 90% of single-order prompts
- Gate: PASS

## Certification (estimate: ramsey_linear_pair, one row per claim)

| Check | Value | Needed | Claims | Pass |
| --- | --- | --- | --- | --- |
| gate_passed | True | pass |  | yes |
| chance_corr | 0.83 | >= 0.7 | 42 | yes |
| chance_mae | 0.14 | <= 0.2 | 42 | yes |
| partition_coherence_p | 1.5e-16 | p >= 0.05 (consistent with exactly 1) | 13 | no |

Described, not tested: fact_accuracy = 0.70 (50 claims). Share of facts where the true version gets the higher credence; describes the model's knowledge, not the measure, so it is not a certification check.

Described, not tested: incoherence_sure_loss = 0.44 (13 partitions). Mean |sum of a partition's cells - 1|: the sure loss per £1 a Dutch book could extract from an agent acting on these numbers (degree of incoherence).

Described, not tested: partitions_above_1 = 0.02 (13 partitions). 11 of 13 partition sums are above 1; two-sided sign-test p-value shown (under coherence, sums scatter evenly either side of 1; no error model needed).

Described, not tested: partitions_within_2sd = 0.77 (13 partitions). Share of partitions whose sum is within 2 measurement-error SDs of 1 (pooled chi-squared 112.2 on 13 df).

## All checks

| Section | Check | qwen3-32b |
| --- | --- | --- |
| Stage 1 | Neutral event N | disc |
|  | Ticket scale (script's verdict) | k/100 |
|  | Dominance, orders averaged | 5 of 5 |
|  | Dominance, single orders | 9 of 10 |
|  | Tickets vs equal stated chance (mean p) | 0.59 |
|  | Transitivity: cyclic triples | 0 of 20 |
|  | Sure amount chosen over a bet | 1.00 |
|  | Pairs agreeing with expected value | 0.43 |
| Utility scale | m(1/2), £ | 4.40 |
|  | Points found | 4 of 6 |
|  | Power-curve rho (1 = linear) | 0.09 |
| Matching | Clean crossing | 0.59 |
|  | Censored (never crosses) | 0.00 |
|  | Mean b(P) + b(not P) | 0.43 |
|  | Chance claims: correlation with true chance | 0.59 |
|  | Chance claims: mean absolute error | 0.24 |
|  | Facts: accuracy | 0.54 |
|  | Forecasts: correlation with crowd | -0.04 |
| Ramsey, linear utility | Clean crossing | 0.53 |
|  | Censored (never crosses) | 0.09 |
|  | Mean b(P) + b(not P) | 0.68 |
|  | Chance claims: correlation with true chance | 0.73 |
|  | Chance claims: mean absolute error | 0.22 |
|  | Facts: accuracy | 0.63 |
|  | Forecasts: correlation with crowd | -0.07 |
| Ramsey, measured utility | Clean crossing | 0.53 |
|  | Censored (never crosses) | 0.09 |
|  | Mean b(P) + b(not P) | 1.61 |
|  | Chance claims: correlation with true chance | 0.73 |
|  | Chance claims: mean absolute error | 0.42 |
|  | Facts: accuracy | 0.56 |
|  | Forecasts: correlation with crowd | -0.01 |
| Matching, pair-corrected | Chance claims: correlation with true chance | 0.75 |
|  | Chance claims: mean absolute error | 0.17 |
|  | Facts: accuracy | 0.66 |
|  | Forecasts: correlation with crowd | -0.07 |
| Ramsey, pair-corrected | Chance claims: correlation with true chance | 0.82 |
|  | Chance claims: mean absolute error | 0.14 |
|  | Facts: accuracy | 0.70 |
|  | Forecasts: correlation with crowd | -0.14 |
| Ramsey (measured utility), pair-corrected | Chance claims: correlation with true chance | 0.82 |
|  | Chance claims: mean absolute error | 0.20 |
|  | Facts: accuracy | 0.70 |
|  | Forecasts: correlation with crowd | -0.01 |
