# LoRA Fine-Tuning of Llama 3.2 (1B) for Spam Detection and Generation

Parameter-efficient fine-tuning of **Meta Llama-3.2-1B** on the
[Deysi/spam-detection-dataset](https://huggingface.co/datasets/Deysi/spam-detection-dataset) so that a
single small model can do two jobs:

1. **Detect**: answer `spam` / `not_spam` for a given message.
2. **Generate**: produce realistic spam examples (useful for stress-testing a filter).

Built for the Bayesian Machine Learning course (TU Delft MSc).

## Method

| Component | Choice |
| --- | --- |
| Base model | `meta-llama/Llama-3.2-1B`, loaded in 8-bit (bitsandbytes) |
| Adapter | LoRA on `q_proj`, `v_proj`; r = 128, α = 256, dropout 0.05 |
| Data | Each message becomes a *classify* instruction; spam messages also become a *generate* instruction |
| Objective | Causal-LM loss on the answer tokens only (prompt and padding masked to −100) |
| Training | 3 epochs, lr 1e-4, fp16, max length 128, 100 held-out examples for evaluation |

## Running

Llama 3.2 is a gated model. Request access on Hugging Face, then:

```bash
pip install -r requirements.txt
export HF_TOKEN=<your Hugging Face token>
python Llama_ft_v2.py
```

The adapter is saved to `./Llama_ft/`. The script ends by generating a spam message and asking the
fine-tuned model to classify it.

A CUDA GPU is required for 8-bit loading.

## Notes

- The label construction was corrected so that the loss is computed on the answer tokens (see the
  `_preprocess_batch` docstring). Results from earlier runs used the previous version.
