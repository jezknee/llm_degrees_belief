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
| `reasoning` | Thinking on: `samples`, `max_tokens`, `temperature`, `top_p`, `top_k`, `claims` (cut-down set), `every` (grid thinning) | 5, 3000, 0.6, 0.95, 20, 20, 2 |
| `neutral` | Neutral event for the utility scale, or `auto` (Stage 1 picks it) | `auto` |
| `certify` | Certification thresholds: `estimate`, `min_chance_corr`, `max_chance_mae`, `min_fact_accuracy` | `ramsey_linear_pair`, 0.7, 0.2, 0.6 |

Models without a thinking switch (Mistral, Llama) need `"thinking_switch": null`, or the
switch is passed to a chat template that doesn't expect it.
