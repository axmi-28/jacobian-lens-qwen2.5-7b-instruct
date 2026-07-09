# Jacobian Lens on Qwen2.5-7B-Instruct

Fitting and applying the **Jacobian lens** (J-lens) from Anthropic's
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
to `Qwen/Qwen2.5-7B-Instruct`, using the reference implementation
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (vendored in
[jacobian-lens/](jacobian-lens/)).

## scripts/slice_repl.py for inference

Interactive explorer that loads model and lens, then allows for typed prompt:

```
prompt> Fact: The capital of the country whose flag has a red maple leaf is
prompt> !gen 30 Q: Which is heavier, a kg of feathers or a kg of steel? A:
prompt> !pin lira Rome     # force rank-tracking for words never in any top-10 cell
prompt> !raw / !stride 2 / q
```


## Published lens

The fitted lens is on the Hugging Face Hub with a model card:
[dormantx/jacobian-lens-qwen2.5-7b-instruct](https://huggingface.co/dormantx/jacobian-lens-qwen2.5-7b-instruct)
(`qwen7b_lens.pt`, 0.65 GiB fp16 — layers 0–26, 100 WikiText prompts, dim_batch=64).

## Layout

```
jacobian-lens/     vendored reference implementation (upstream clone)
.venv/             fresh Python 3.14 venv (torch 2.13 MPS, transformers 5.13)
scripts/           probe / fit / apply scripts written for this project
out/               fitted lens, checkpoints, HTML slice (created at GATE 1+)
```
