# Report card: Qwen/Qwen3-32B (thinking on)

Run `qwen3-32b-thinking`, 2026-11-14. **NOT CERTIFIED: see the failed checks**

## Gate (smoke test and Stage 1)

- matching: dominance at the top of the grid 85%, curves crossing 0.5 70%, mean order gap 0.06
- Stage 1 dominance: 100% of single-order prompts
- Gate: FAIL

## Certification (estimate: ramsey_linear_pair, one row per claim)

| Check | Value | Needed | Claims | Pass |
| --- | --- | --- | --- | --- |
| gate_passed | False | pass |  | no |
| chance_corr | 1.00 | >= 0.7 | 6 | yes |
| chance_mae | 0.01 | <= 0.2 | 6 | yes |
| partition_coherence_p | n/a | p >= 0.05 (consistent with exactly 1) | 0 | no |

Described, not tested: fact_accuracy = 1.00 (3 claims). Share of facts where the true version gets the higher credence; describes the model's knowledge, not the measure, so it is not a certification check.

Described, not tested: incoherence_sure_loss = n/a (0 partitions). Mean |sum of a partition's cells - 1|: the sure loss per £1 a Dutch book could extract from an agent acting on these numbers (degree of incoherence).

Described, not tested: partitions_above_1 = n/a (0 partitions). No partitions.

Described, not tested: partitions_within_2sd = n/a (0 partitions). No partitions.

## All checks

| Section | Check | qwen3-32b-thinking |
| --- | --- | --- |
| Stage 1 | Neutral event N | disc |
|  | Ticket scale (script's verdict) | k/100 |
|  | Dominance, orders averaged | 5 of 5 |
|  | Dominance, single orders | 10 of 10 |
|  | Tickets vs equal stated chance (mean p) | 0.83 |
|  | Transitivity: cyclic triples | 0 of 20 |
|  | Sure amount chosen over a bet | 0.54 |
|  | Pairs agreeing with expected value | 1.00 |
| Matching | Clean crossing | 0.60 |
|  | Censored (never crosses) | 0.30 |
|  | Mean b(P) + b(not P) | 0.79 |
|  | Chance claims: correlation with true chance | 1.00 |
|  | Chance claims: mean absolute error | 0.03 |
|  | Facts: accuracy | 0.50 |
|  | Forecasts: correlation with crowd | 0.09 |
| Ramsey, linear utility | Clean crossing | 0.60 |
|  | Censored (never crosses) | 0.28 |
|  | Mean b(P) + b(not P) | 0.73 |
|  | Chance claims: correlation with true chance | 1.00 |
|  | Chance claims: mean absolute error | 0.03 |
|  | Facts: accuracy | 0.71 |
|  | Forecasts: correlation with crowd | -0.10 |
| Ramsey, measured utility | Clean crossing | n/a |
|  | Censored (never crosses) | n/a |
|  | Mean b(P) + b(not P) | n/a |
|  | Chance claims: correlation with true chance | n/a |
|  | Chance claims: mean absolute error | n/a |
|  | Facts: accuracy | n/a |
|  | Forecasts: correlation with crowd | n/a |
| Matching, pair-corrected | Chance claims: correlation with true chance | 1.00 |
|  | Chance claims: mean absolute error | 0.01 |
|  | Facts: accuracy | 0.75 |
|  | Forecasts: correlation with crowd | 0.11 |
| Ramsey, pair-corrected | Chance claims: correlation with true chance | 1.00 |
|  | Chance claims: mean absolute error | 0.01 |
|  | Facts: accuracy | 0.83 |
|  | Forecasts: correlation with crowd | -0.12 |
| Ramsey (measured utility), pair-corrected | Chance claims: correlation with true chance | n/a |
|  | Chance claims: mean absolute error | n/a |
|  | Facts: accuracy | n/a |
|  | Forecasts: correlation with crowd | n/a |
