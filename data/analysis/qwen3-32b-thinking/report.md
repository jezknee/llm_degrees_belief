# Credence analysis: qwen3-32b-thinking

Credence = where the order-averaged choice curve crosses 0.5. Censored curves count at the end of the grid (see status).

## Well-defined and coherent?

| Estimate | Propositions | Clean crossing | Censored | Curves falling | Mean b(P)+b(not P) | Mean Dutch-book loss |
| --- | --- | --- | --- | --- | --- | --- |
| matching | 40 | 60% | 30% | 85% | 0.79 | 0.29 |
| ramsey_linear | 39 | 60% | 28% | 82% | 0.73 | 0.30 |

Pair-corrected estimates, b*(P) = (b(P) - b(not P) + 1) / 2, sum to 1 by construction; they appear below with the suffix _pair.

## Calibration on the anchors

| Estimate | Chance: MAE vs true chance | Chance: correlation | Facts: Brier | Facts: accuracy | Forecasts: MAE vs crowd | Forecasts: correlation |
| --- | --- | --- | --- | --- | --- | --- |
| matching | 0.03 | 1.00 | 0.37 | 0.50 | 0.33 | 0.09 |
| ramsey_linear | 0.03 | 1.00 | 0.30 | 0.71 | 0.35 | -0.10 |
| matching_pair | 0.01 | 1.00 | 0.28 | 0.75 | 0.21 | 0.11 |
| ramsey_linear_pair | 0.01 | 1.00 | 0.09 | 0.83 | 0.16 | -0.12 |

## Do the methods agree?

| Tier | Compared | With | Propositions | Correlation | Mean difference (Ramsey - matching) |
| --- | --- | --- | --- | --- | --- |
| all | matching | ramsey_linear | 39 | 0.92 | -0.02 |
| all | matching | ramsey_utility | 0 | n/a | n/a |
| all | matching_pair | ramsey_linear_pair | 38 | 0.94 | -0.00 |
| all | matching_pair | ramsey_utility_pair | 0 | n/a | n/a |
| chance | matching | ramsey_linear | 8 | 1.00 | -0.00 |
| chance | matching | ramsey_utility | 0 | n/a | n/a |
| chance | matching_pair | ramsey_linear_pair | 8 | 1.00 | -0.00 |
| chance | matching_pair | ramsey_utility_pair | 0 | n/a | n/a |
| fact | matching | ramsey_linear | 7 | 0.98 | -0.05 |
| fact | matching | ramsey_utility | 0 | n/a | n/a |
| fact | matching_pair | ramsey_linear_pair | 6 | 0.97 | 0.00 |
| fact | matching_pair | ramsey_utility_pair | 0 | n/a | n/a |
| forecast | matching | ramsey_linear | 8 | 0.49 | -0.07 |
| forecast | matching | ramsey_utility | 0 | n/a | n/a |
| forecast | matching_pair | ramsey_linear_pair | 8 | 0.67 | 0.00 |
| forecast | matching_pair | ramsey_utility_pair | 0 | n/a | n/a |
| kuq | matching | ramsey_linear | 8 | 0.96 | -0.02 |
| kuq | matching | ramsey_utility | 0 | n/a | n/a |
| kuq | matching_pair | ramsey_linear_pair | 8 | 0.98 | 0.00 |
| kuq | matching_pair | ramsey_utility_pair | 0 | n/a | n/a |
| partition | matching | ramsey_linear | 8 | 0.85 | 0.06 |
| partition | matching | ramsey_utility | 0 | n/a | n/a |
| partition | matching_pair | ramsey_linear_pair | 8 | 0.92 | -0.00 |
| partition | matching_pair | ramsey_utility_pair | 0 | n/a | n/a |

## Where do Ramsey curves fall?

| Matching credence | Propositions | Ramsey falling | Ramsey never below 0.5 | Mean certainty equivalent |
| --- | --- | --- | --- | --- |
| 0-0.2 | 18 | 0.83 | 0.00 | £50 |
| 0.2-0.4 | 5 | 0.80 | 0.00 | £436 |
| 0.4-0.6 | 3 | 1.00 | 0.00 | £480 |
| 0.6-0.8 | 5 | 0.80 | 0.00 | £483 |
| 0.8-1 | 9 | 0.78 | 0.67 | £940 |
