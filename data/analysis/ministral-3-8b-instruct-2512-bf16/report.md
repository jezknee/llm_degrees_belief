# Credence analysis: ministral-3-8b-instruct-2512-bf16

Credence = where the order-averaged choice curve crosses 0.5. Censored curves count at the end of the grid (see status).

## Well-defined and coherent?

| Estimate | Propositions | Clean crossing | Censored | Curves falling | Mean b(P)+b(not P) | Mean Dutch-book loss |
| --- | --- | --- | --- | --- | --- | --- |
| matching | 638 | 0% | 94% | 100% | 1.92 | 0.92 |
| ramsey_linear | 636 | 0% | 99% | 61% | 1.97 | 0.97 |
| ramsey_utility | 636 | 0% | 99% | 61% | 1.97 | 0.97 |

Pair-corrected estimates, b*(P) = (b(P) - b(not P) + 1) / 2, sum to 1 by construction; they appear below with the suffix _pair.

## Calibration on the anchors

| Estimate | Chance: MAE vs true chance | Chance: correlation | Facts: Brier | Facts: accuracy | Forecasts: MAE vs crowd | Forecasts: correlation |
| --- | --- | --- | --- | --- | --- | --- |
| matching | 0.45 | 0.33 | 0.47 | 0.53 | 0.47 | 0.12 |
| ramsey_linear | 0.48 | 0.16 | 0.47 | 0.53 | 0.50 | n/a |
| ramsey_utility | 0.48 | 0.16 | 0.47 | 0.53 | 0.50 | n/a |
| matching_pair | 0.21 | 0.44 | 0.26 | 0.52 | 0.23 | 0.17 |
| ramsey_linear_pair | 0.24 | 0.23 | 0.26 | 0.52 | 0.22 | n/a |
| ramsey_utility_pair | 0.24 | 0.23 | 0.26 | 0.52 | 0.22 | n/a |

## Do the methods agree?

| Tier | Compared | With | Propositions | Correlation | Mean difference (Ramsey - matching) |
| --- | --- | --- | --- | --- | --- |
| all | matching | ramsey_linear | 636 | 0.22 | 0.03 |
| all | matching | ramsey_utility | 636 | 0.23 | 0.03 |
| all | matching_pair | ramsey_linear_pair | 634 | 0.25 | -0.00 |
| all | matching_pair | ramsey_utility_pair | 634 | 0.26 | -0.00 |
| chance | matching | ramsey_linear | 60 | -0.05 | 0.02 |
| chance | matching | ramsey_utility | 60 | -0.05 | 0.02 |
| chance | matching_pair | ramsey_linear_pair | 60 | 0.00 | 0.00 |
| chance | matching_pair | ramsey_utility_pair | 60 | 0.00 | 0.00 |
| fact | matching | ramsey_linear | 98 | 0.55 | -0.02 |
| fact | matching | ramsey_utility | 98 | 0.55 | -0.02 |
| fact | matching_pair | ramsey_linear_pair | 96 | 0.58 | 0.00 |
| fact | matching_pair | ramsey_utility_pair | 96 | 0.58 | 0.00 |
| forecast | matching | ramsey_linear | 240 | n/a | 0.07 |
| forecast | matching | ramsey_utility | 240 | n/a | 0.07 |
| forecast | matching_pair | ramsey_linear_pair | 240 | n/a | 0.00 |
| forecast | matching_pair | ramsey_utility_pair | 240 | n/a | 0.00 |
| kuq | matching | ramsey_linear | 160 | n/a | 0.01 |
| kuq | matching | ramsey_utility | 160 | n/a | 0.01 |
| kuq | matching_pair | ramsey_linear_pair | 160 | n/a | 0.00 |
| kuq | matching_pair | ramsey_utility_pair | 160 | n/a | 0.00 |
| partition | matching | ramsey_linear | 78 | 1.00 | 0.00 |
| partition | matching | ramsey_utility | 78 | 1.00 | -0.00 |
| partition | matching_pair | ramsey_linear_pair | 78 | 1.00 | -0.00 |
| partition | matching_pair | ramsey_utility_pair | 78 | 1.00 | -0.00 |

## Where do Ramsey curves fall?

| Matching credence | Propositions | Ramsey falling | Ramsey never below 0.5 | Mean certainty equivalent |
| --- | --- | --- | --- | --- |
| 0-0.2 | 6 | 0.17 | 0.67 | £683 |
| 0.2-0.4 | 28 | 0.54 | 0.86 | £938 |
| 0.4-0.6 | 1 | 0.00 | 1.00 | £1000 |
| 0.6-0.8 | 4 | 0.50 | 1.00 | £1000 |
| 0.8-1 | 599 | 0.62 | 0.99 | £992 |

## Utility scale

u(£500) = 0.064, u(£900) = 0.114, u(£990) = 0.285 (linear would be 0.5, 0.9, 0.99).
