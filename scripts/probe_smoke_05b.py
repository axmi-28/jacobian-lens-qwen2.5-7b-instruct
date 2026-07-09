"""GATE 0 smoke test: fit jlens on Qwen2.5-0.5B-Instruct on MPS, 2 prompts.

Validates the full pipeline locally and yields an MPS throughput datapoint
for extrapolating 7B feasibility.
"""

import logging
import time

import torch
import transformers

import jlens
import jlens.examples

logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "mps"

t0 = time.perf_counter()
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
hf = hf.to(DEVICE)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
print(f"model loaded in {time.perf_counter() - t0:.1f}s")

model = jlens.from_hf(hf, tok)
print("repr:", repr(model))
print("n_layers:", model.n_layers, "d_model:", model.d_model)

prompts = jlens.examples.load_wikitext_prompts(2)
print("prompt lens (chars):", [len(p) for p in prompts])

t0 = time.perf_counter()
lens = jlens.fit(model, prompts, dim_batch=8, max_seq_len=128)
elapsed = time.perf_counter() - t0
print(f"\nfit: {elapsed:.1f}s total, {elapsed / 2:.1f}s/prompt")
print(f"peak MPS mem: {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB driver")

# quick apply sanity check
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-1], layers=[model.n_layers // 2],
)
mid = model.n_layers // 2
top = lens_logits[mid][0].topk(5).indices
print("mid-layer top5:", [tok.decode([t]) for t in top])
