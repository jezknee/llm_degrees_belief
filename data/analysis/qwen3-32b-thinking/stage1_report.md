# Stage 1 results: qwen3-32b-thinking

Indifference tolerance: |p - 0.5| <= 0.1. p = probability of choosing the target option, averaged over both orders.

## 1. Neutral event

- disc: mean p 0.50, worst pair off by 0.00, mean |order gap| 0.83, vs stated 50%: 0.17 (indifferent for every pair)
- die: mean p 0.53, worst pair off by 0.08, mean |order gap| 0.78, vs stated 50%: 0.33 (indifferent for every pair)
- coin: mean p 0.67, worst pair off by 0.42, mean |order gap| 0.50, vs stated 50%: 0.92

**N = disc** (indifferent for every outcome pair).

## 2. Ticket symmetry

- Single tickets: 30 pairs, 100% indifferent; mean p of the lower-numbered ticket 0.50; correlation with distance n/a.
- Blocks: symmetry_block_1_10_vs_91_100 0.5, symmetry_block_1_25_vs_26_50 0.5, symmetry_block_1_50_vs_51_100 0.5.

**Ticket scale: k/100** (tickets treated as exchangeable: read ticket bets as k/100).

## 3. Validation: tickets vs stated chances

- Mean p(tickets) 0.83 (range 0.58 to 0.92); correlation with k 0.09.
- Reading: the model prefers tickets to an equal stated chance. A roughly constant preference across k is a framing effect, not a distortion of the ticket scale.

## 4. Dominance

- 5 of 5 correct after averaging orders; 10 of 10 single-order prompts correct.
- Failures: none.

## 5. Transitivity

- 0 of 20 triples cyclic.
- 100% of pairs with different expected values prefer the higher one.
- Sure amount vs bet: the sure amount is chosen in 54% of such pairs.
- Pairs asked in more than one triple: 21; answered inconsistently: 0.

## Position bias

| Step | Mean order gap | Share flipping with order |
| --- | --- | --- |
| dominance | +0.00 | 0% |
| neutral | +0.70 | 89% |
| symmetry_block | +0.83 | 100% |
| symmetry_single | +0.83 | 100% |
| transitivity | -0.03 | 2% |
| validation_neutral | -0.11 | 0% |
| validation_tickets | -0.04 | 0% |
