# Report card: mistralai/Ministral-3-8B-Instruct-2512-BF16

Run `ministral-3-8b-instruct-2512-bf16`, 2026-11-14. **NOT CERTIFIED: see the failed checks**

## Gate (smoke test and Stage 1)

- matching: dominance at the top of the grid 0%, curves crossing 0.5 0%, mean order gap 0.07
- ramsey: dominance at the top of the grid 0%, curves crossing 0.5 0%, mean order gap 0.15
- Stage 1 dominance: 90% of single-order prompts
- Gate: FAIL

## Certification (estimate: ramsey_linear_pair, one row per claim)

| Check | Value | Needed | Claims | Pass |
| --- | --- | --- | --- | --- |
| gate_passed | False | pass |  | no |
| chance_corr | 0.38 | >= 0.7 | 42 | no |
| chance_mae | 0.21 | <= 0.2 | 42 | no |
| partition_coherence_p | < 1e-16 | p >= 0.05 (consistent with exactly 1) | 13 | no |

Described, not tested: fact_accuracy = 0.48 (48 claims). Share of facts where the true version gets the higher credence; describes the model's knowledge, not the measure, so it is not a certification check.

Described, not tested: incoherence_sure_loss = 0.47 (13 partitions). Mean |sum of a partition's cells - 1|: the sure loss per £1 a Dutch book could extract from an agent acting on these numbers (degree of incoherence).

Described, not tested: partitions_above_1 = 2.4e-04 (13 partitions). 13 of 13 partition sums are above 1; two-sided sign-test p-value shown (under coherence, sums scatter evenly either side of 1; no error model needed).

Described, not tested: partitions_within_2sd = 0.15 (13 partitions). Share of partitions whose sum is within 2 measurement-error SDs of 1 (pooled chi-squared 7007.5 on 13 df).

## All checks

| Section | Check | ministral-3-8b-instruct-2512-bf16 |
| --- | --- | --- |
| Stage 1 | Neutral event N | die |
|  | Ticket scale (script's verdict) | measured |
|  | Dominance, orders averaged | 5 of 5 |
|  | Dominance, single orders | 9 of 10 |
|  | Tickets vs equal stated chance (mean p) | 0.71 |
|  | Transitivity: cyclic triples | 0 of 20 |
|  | Sure amount chosen over a bet | 0.00 |
|  | Pairs agreeing with expected value | 0.41 |
| Utility scale | m(1/2), £ | 993.90 |
|  | Points found | 6 of 6 |
|  | Power-curve rho (1 = linear) | 123.26 |
| Matching | Clean crossing | 0.00 |
|  | Censored (never crosses) | 0.94 |
|  | Mean b(P) + b(not P) | 1.92 |
|  | Chance claims: correlation with true chance | 0.33 |
|  | Chance claims: mean absolute error | 0.45 |
|  | Facts: accuracy | 0.53 |
|  | Forecasts: correlation with crowd | 0.12 |
| Ramsey, linear utility | Clean crossing | 0.00 |
|  | Censored (never crosses) | 0.99 |
|  | Mean b(P) + b(not P) | 1.97 |
|  | Chance claims: correlation with true chance | 0.16 |
|  | Chance claims: mean absolute error | 0.48 |
|  | Facts: accuracy | 0.53 |
|  | Forecasts: correlation with crowd | n/a |
| Ramsey, measured utility | Clean crossing | 0.00 |
|  | Censored (never crosses) | 0.99 |
|  | Mean b(P) + b(not P) | 1.97 |
|  | Chance claims: correlation with true chance | 0.16 |
|  | Chance claims: mean absolute error | 0.48 |
|  | Facts: accuracy | 0.53 |
|  | Forecasts: correlation with crowd | n/a |
| Matching, pair-corrected | Chance claims: correlation with true chance | 0.44 |
|  | Chance claims: mean absolute error | 0.21 |
|  | Facts: accuracy | 0.52 |
|  | Forecasts: correlation with crowd | 0.17 |
| Ramsey, pair-corrected | Chance claims: correlation with true chance | 0.23 |
|  | Chance claims: mean absolute error | 0.24 |
|  | Facts: accuracy | 0.52 |
|  | Forecasts: correlation with crowd | n/a |
| Ramsey (measured utility), pair-corrected | Chance claims: correlation with true chance | 0.23 |
|  | Chance claims: mean absolute error | 0.24 |
|  | Facts: accuracy | 0.52 |
|  | Forecasts: correlation with crowd | n/a |
