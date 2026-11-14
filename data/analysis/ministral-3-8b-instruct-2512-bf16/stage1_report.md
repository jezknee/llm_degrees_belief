# Stage 1 results: ministral-3-8b-instruct-2512-bf16

Indifference tolerance: |p - 0.5| <= 0.1. p = probability of choosing the target option, averaged over both orders.

## 1. Neutral event

- die: mean p 0.51, worst pair off by 0.02, mean |order gap| 0.25, vs stated 50%: 0.71 (indifferent for every pair)
- disc: mean p 0.38, worst pair off by 0.14, mean |order gap| 0.08, vs stated 50%: 0.42
- coin: mean p 0.69, worst pair off by 0.23, mean |order gap| 0.17, vs stated 50%: 0.79

**N = die** (indifferent for every outcome pair).

## 2. Ticket symmetry

- Single tickets: 30 pairs, 67% indifferent; mean p of the lower-numbered ticket 0.45; correlation with distance 0.35.
- Blocks: symmetry_block_1_10_vs_91_100 0.355, symmetry_block_1_25_vs_26_50 0.47, symmetry_block_1_50_vs_51_100 0.516.

**Ticket scale: measured** (tickets not treated as exchangeable: matching needs a measured reference curve).

## 3. Validation: tickets vs stated chances

- Mean p(tickets) 0.71 (range 0.65 to 0.76); correlation with k 0.31.
- Reading: the model prefers tickets to an equal stated chance. A roughly constant preference across k is a framing effect, not a distortion of the ticket scale.

## 4. Dominance

- 5 of 5 correct after averaging orders; 9 of 10 single-order prompts correct.
- Failures: none.

## 5. Transitivity

- 0 of 20 triples cyclic.
- 41% of pairs with different expected values prefer the higher one.
- Sure amount vs bet: the sure amount is chosen in 0% of such pairs.
- Pairs asked in more than one triple: 21; answered inconsistently: 0.

## Position bias

| Step | Mean order gap | Share flipping with order |
| --- | --- | --- |
| dominance | +0.15 | 20% |
| neutral | +0.14 | 44% |
| symmetry_block | +0.72 | 100% |
| symmetry_single | +0.05 | 40% |
| transitivity | +0.18 | 2% |
| validation_neutral | -0.07 | 0% |
| validation_tickets | +0.04 | 0% |
