# Stage 1 results: qwen3-32b

Indifference tolerance: |p - 0.5| <= 0.1. p = probability of choosing the target option, averaged over both orders.

## 1. Neutral event

- disc: mean p 0.50, worst pair off by 0.00, mean |order gap| 1.00, vs stated 50%: 0.47 (indifferent for every pair)
- die: mean p 0.51, worst pair off by 0.02, mean |order gap| 0.98, vs stated 50%: 0.59 (indifferent for every pair)
- coin: mean p 0.52, worst pair off by 0.07, mean |order gap| 0.95, vs stated 50%: 0.94 (indifferent for every pair)

**N = disc** (indifferent for every outcome pair).

## 2. Ticket symmetry

- Single tickets: 30 pairs, 100% indifferent; mean p of the lower-numbered ticket 0.50; correlation with distance 0.38.
- Blocks: symmetry_block_1_10_vs_91_100 0.591, symmetry_block_1_25_vs_26_50 0.5, symmetry_block_1_50_vs_51_100 0.5.

**Ticket scale: k/100** (tickets treated as exchangeable: read ticket bets as k/100).

## 3. Validation: tickets vs stated chances

- Mean p(tickets) 0.59 (range 0.50 to 0.71); correlation with k -0.22.
- Reading: the model prefers tickets to an equal stated chance. A roughly constant preference across k is a framing effect, not a distortion of the ticket scale.

## 4. Dominance

- 5 of 5 correct after averaging orders; 9 of 10 single-order prompts correct.
- Failures: none.

## 5. Transitivity

- 0 of 20 triples cyclic.
- 43% of pairs with different expected values prefer the higher one.
- Sure amount vs bet: the sure amount is chosen in 100% of such pairs.
- Pairs asked in more than one triple: 21; answered inconsistently: 0.

## Position bias

| Step | Mean order gap | Share flipping with order |
| --- | --- | --- |
| dominance | +0.16 | 20% |
| neutral | +0.98 | 100% |
| symmetry_block | +0.94 | 100% |
| symmetry_single | +0.94 | 100% |
| transitivity | +0.20 | 47% |
| validation_neutral | +0.49 | 67% |
| validation_tickets | +0.79 | 100% |
