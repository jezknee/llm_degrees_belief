# Runs compared

| Section | Check | ministral-3-8b-instruct-2512-bf16 | qwen3-32b | qwen3-32b-thinking |
| --- | --- | --- | --- | --- |
| Stage 1 | Neutral event N | die | disc | disc |
|  | Ticket scale (script's verdict) | measured | k/100 | k/100 |
|  | Dominance, orders averaged | 5 of 5 | 5 of 5 | 5 of 5 |
|  | Dominance, single orders | 9 of 10 | 9 of 10 | 10 of 10 |
|  | Tickets vs equal stated chance (mean p) | 0.71 | 0.59 | 0.83 |
|  | Transitivity: cyclic triples | 0 of 20 | 0 of 20 | 0 of 20 |
|  | Sure amount chosen over a bet | 0.00 | 1.00 | 0.54 |
|  | Pairs agreeing with expected value | 0.41 | 0.43 | 1.00 |
| Utility scale | m(1/2), £ | 993.90 | 4.40 | n/a |
|  | Points found | 6 of 6 | 4 of 6 | n/a |
|  | Power-curve rho (1 = linear) | 123.26 | 0.09 | n/a |
| Matching | Clean crossing | 0.00 | 0.59 | 0.60 |
|  | Censored (never crosses) | 0.94 | 0.00 | 0.30 |
|  | Mean b(P) + b(not P) | 1.92 | 0.43 | 0.79 |
|  | Chance claims: correlation with true chance | 0.33 | 0.59 | 1.00 |
|  | Chance claims: mean absolute error | 0.45 | 0.24 | 0.03 |
|  | Facts: accuracy | 0.53 | 0.54 | 0.50 |
|  | Forecasts: correlation with crowd | 0.12 | -0.04 | 0.09 |
| Ramsey, linear utility | Clean crossing | 0.00 | 0.53 | 0.60 |
|  | Censored (never crosses) | 0.99 | 0.09 | 0.28 |
|  | Mean b(P) + b(not P) | 1.97 | 0.68 | 0.73 |
|  | Chance claims: correlation with true chance | 0.16 | 0.73 | 1.00 |
|  | Chance claims: mean absolute error | 0.48 | 0.22 | 0.03 |
|  | Facts: accuracy | 0.53 | 0.63 | 0.71 |
|  | Forecasts: correlation with crowd | n/a | -0.07 | -0.10 |
| Ramsey, measured utility | Clean crossing | 0.00 | 0.53 | n/a |
|  | Censored (never crosses) | 0.99 | 0.09 | n/a |
|  | Mean b(P) + b(not P) | 1.97 | 1.61 | n/a |
|  | Chance claims: correlation with true chance | 0.16 | 0.73 | n/a |
|  | Chance claims: mean absolute error | 0.48 | 0.42 | n/a |
|  | Facts: accuracy | 0.53 | 0.56 | n/a |
|  | Forecasts: correlation with crowd | n/a | -0.01 | n/a |
| Matching, pair-corrected | Chance claims: correlation with true chance | 0.44 | 0.75 | 1.00 |
|  | Chance claims: mean absolute error | 0.21 | 0.17 | 0.01 |
|  | Facts: accuracy | 0.52 | 0.66 | 0.75 |
|  | Forecasts: correlation with crowd | 0.17 | -0.07 | 0.11 |
| Ramsey, pair-corrected | Chance claims: correlation with true chance | 0.23 | 0.82 | 1.00 |
|  | Chance claims: mean absolute error | 0.24 | 0.14 | 0.01 |
|  | Facts: accuracy | 0.52 | 0.70 | 0.83 |
|  | Forecasts: correlation with crowd | n/a | -0.14 | -0.12 |
| Ramsey (measured utility), pair-corrected | Chance claims: correlation with true chance | 0.23 | 0.82 | n/a |
|  | Chance claims: mean absolute error | 0.24 | 0.20 | n/a |
|  | Facts: accuracy | 0.52 | 0.70 | n/a |
|  | Forecasts: correlation with crowd | n/a | -0.01 | n/a |
