# Model configs for credence_pipeline.py

One JSON file per model (and per thinking setting). Only `model` is required.

| Key | Meaning | Default |
| --- | --- | --- |
| `model` | Hugging Face repo id | required |
| `name` | Run name used in file names | short form of `model` (`-thinking` added when thinking is on) |
| `backend` | `vllm`, or `fake` for testing without a GPU | `vllm` |
| `thinking` | Reason before answering (sampled), or answer at once (log-probabilities) | `false` |
| `thinking_switch` | Chat-template switch for thinking (Qwen: `enable_thinking`); `null` for models without one | `enable_thinking` |
| `chat_template_kwargs` | Any other chat-template options | none |
| `vllm` | Passed straight to vLLM's `LLM(...)`: `dtype`, `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`, `tokenizer_mode`, ... | none |
| `top_k` | Top tokens read per answer (thinking off) | 20 |
| `reasoning` | Thinking on: `samples`, `max_tokens`, `temperature`, `top_p`, `top_k`, `claims` (cut-down set), `partitions` (whole partitions kept: `all` or a number), `every` (grid thinning) | 5, 3000, 0.6, 0.95, 20, 20, all, 2 |
| `neutral` | Neutral event for the utility scale, or `auto` (Stage 1 picks it) | `auto` |
| `certify` | Certification thresholds: `estimate`, `min_chance_corr`, `max_chance_mae`, `partition_alpha` | `ramsey_linear_pair`, 0.7, 0.2, 0.05 |

Models without a thinking switch (Mistral, Llama) need `"thinking_switch": null`, or the
switch is passed to a chat template that doesn't expect it.

Certification checks the measure, not the model's knowledge: chance claims (whose right answer the
prompt fixes) and partition coherence. Coherence is exact (a partition's three cells must sum to exactly
1), so the only tolerance is for measurement error: each sum's error is estimated from the grid step and
half the spread between the two option orders, and the sums must be consistent with 1 (pooled
chi-squared test, p >= partition_alpha). The degree of incoherence (sure loss per £1 to a Dutch book) and
facts accuracy are reported but not tested.

## `credence_tool.py` (version 2: screen, then measure)

`screen` runs a small fixed battery instead of every claim: Stage 1, about 12 chance claims spread
across the probability range, every chance partition and a few fact partitions (all with negations),
plus the utility scale, on a thinned grid. It stops early if the chance partitions already fail
coherence decisively. `measure` then does the weighty work on your own claims: full grid, both orders,
more samples. Optional `screen` block in a config:

| Key | Meaning | Default |
| --- | --- | --- |
| `chance_claims` | Chance claims in the battery, spread across the true-chance range | 12 |
| `fact_partitions` | Whole fact partitions in the battery (all chance partitions are always included) | 4 |
| `every` | Grid thinning for the screen (2 = every other value) | 2 |
| `samples` | Samples per prompt in the screen, thinking on | 3 |
| `early_stop_p` | Stop after the chance partitions if coherence p is below this | 0.001 |
| `measure_samples` | Samples per prompt in `measure`, thinking on | 5 |
