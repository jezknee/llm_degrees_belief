# Credence analysis: qwen3-32b

Credence = where the order-averaged choice curve crosses 0.5. Censored curves count at the end of the grid (see status).

## Well-defined and coherent?

| Estimate | Propositions | Clean crossing | Censored | Curves falling | Mean b(P)+b(not P) | Mean Dutch-book loss |
| --- | --- | --- | --- | --- | --- | --- |
| matching | 638 | 59% | 0% | 100% | 0.43 | 0.63 |
| ramsey_linear | 638 | 53% | 9% | 97% | 0.68 | 0.65 |
| ramsey_utility | 638 | 53% | 9% | 97% | 1.61 | 0.64 |

Pair-corrected estimates, b*(P) = (b(P) - b(not P) + 1) / 2, sum to 1 by construction; they appear below with the suffix _pair.

## Calibration on the anchors

| Estimate | Chance: MAE vs true chance | Chance: correlation | Facts: Brier | Facts: accuracy | Forecasts: MAE vs crowd | Forecasts: correlation |
| --- | --- | --- | --- | --- | --- | --- |
| matching | 0.24 | 0.59 | 0.36 | 0.54 | 0.38 | -0.04 |
| ramsey_linear | 0.22 | 0.73 | 0.30 | 0.63 | 0.46 | -0.07 |
| ramsey_utility | 0.42 | 0.73 | 0.34 | 0.56 | 0.32 | -0.01 |
| matching_pair | 0.17 | 0.75 | 0.23 | 0.66 | 0.26 | -0.07 |
| ramsey_linear_pair | 0.14 | 0.82 | 0.20 | 0.70 | 0.26 | -0.14 |
| ramsey_utility_pair | 0.20 | 0.82 | 0.20 | 0.70 | 0.24 | -0.01 |

## Do the methods agree?

| Tier | Compared | With | Propositions | Correlation | Mean difference (Ramsey - matching) |
| --- | --- | --- | --- | --- | --- |
| all | matching | ramsey_linear | 638 | 0.48 | 0.13 |
| all | matching | ramsey_utility | 638 | 0.37 | 0.59 |
| all | matching_pair | ramsey_linear_pair | 638 | 0.48 | -0.00 |
| all | matching_pair | ramsey_utility_pair | 638 | 0.40 | 0.00 |
| chance | matching | ramsey_linear | 60 | 0.49 | 0.35 |
| chance | matching | ramsey_utility | 60 | 0.49 | 0.60 |
| chance | matching_pair | ramsey_linear_pair | 60 | 0.76 | -0.00 |
| chance | matching_pair | ramsey_utility_pair | 60 | 0.76 | -0.00 |
| fact | matching | ramsey_linear | 100 | 0.53 | 0.28 |
| fact | matching | ramsey_utility | 100 | 0.38 | 0.60 |
| fact | matching_pair | ramsey_linear_pair | 100 | 0.60 | 0.00 |
| fact | matching_pair | ramsey_utility_pair | 100 | 0.62 | 0.00 |
| forecast | matching | ramsey_linear | 240 | 0.56 | -0.07 |
| forecast | matching | ramsey_utility | 240 | 0.38 | 0.52 |
| forecast | matching_pair | ramsey_linear_pair | 240 | 0.27 | 0.00 |
| forecast | matching_pair | ramsey_utility_pair | 240 | 0.18 | 0.00 |
| kuq | matching | ramsey_linear | 160 | 0.50 | 0.23 |
| kuq | matching | ramsey_utility | 160 | 0.41 | 0.68 |
| kuq | matching_pair | ramsey_linear_pair | 160 | 0.46 | -0.00 |
| kuq | matching_pair | ramsey_utility_pair | 160 | 0.47 | -0.00 |
| partition | matching | ramsey_linear | 78 | 0.59 | 0.15 |
| partition | matching | ramsey_utility | 78 | 0.46 | 0.63 |
| partition | matching_pair | ramsey_linear_pair | 78 | 0.48 | -0.00 |
| partition | matching_pair | ramsey_utility_pair | 78 | 0.44 | -0.00 |

## Where do Ramsey curves fall?

| Matching credence | Propositions | Ramsey falling | Ramsey never below 0.5 | Mean certainty equivalent |
| --- | --- | --- | --- | --- |
| 0-0.2 | 430 | 0.97 | 0.02 | £215 |
| 0.2-0.4 | 54 | 1.00 | 0.07 | £541 |
| 0.4-0.6 | 112 | 0.99 | 0.11 | £530 |
| 0.6-0.8 | 14 | 1.00 | 0.29 | £841 |
| 0.8-1 | 28 | 0.96 | 0.25 | £829 |

## Utility scale

u(£500) = 0.874, u(£900) = 0.975, u(£990) = 0.997 (linear would be 0.5, 0.9, 0.99).
