"""Runs ON the RunPod H100: GATE 0 probe, then (if --fit) GATE 1 fit.

Stage A (default): load Qwen2.5-7B-Instruct, verify layout, time a 2-prompt
fit at dim_batch=8 over all layers, then a 1-prompt timing at candidate
dim_batch values; write results to out/probe.json and stop.

Stage B (--fit --dim-batch N --n-prompts M): full fit with checkpointing,
save lens to out/qwen7b_lens.pt.
"""

import argparse
import json
import logging
import math
import os
import time

import torch
import transformers

import jlens
import jlens.examples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("remote")

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = "out"


def gib(x: int) -> float:
    return x / 2**30


def load_model():
    t0 = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0"
    )
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    log.info("loaded %s in %.1fs: %r", MODEL, time.perf_counter() - t0, model)
    assert model.n_layers == 28 and model.d_model == 3584, (
        model.n_layers,
        model.d_model,
    )
    return model, tok


def probe(model) -> dict:
    prompts = jlens.examples.load_wikitext_prompts(8, min_chars=600)
    results = {"n_layers": model.n_layers, "d_model": model.d_model, "timings": {}}

    # Guideline probe: 2 prompts, dim_batch=8, ALL layers.
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    jlens.fit(model, prompts[:2], dim_batch=8, max_seq_len=128)
    dt = (time.perf_counter() - t0) / 2
    results["timings"]["8"] = {
        "sec_per_prompt": round(dt, 1),
        "peak_vram_gib": round(gib(torch.cuda.max_memory_allocated()), 2),
    }
    log.info("dim_batch=8: %.1fs/prompt", dt)

    # Candidate larger dim_batch values, 1 prompt each (fresh prompt each time).
    for i, db in enumerate((32, 64)):
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            jlens.fit(model, [prompts[2 + i]], dim_batch=db, max_seq_len=128)
            dt = time.perf_counter() - t0
            results["timings"][str(db)] = {
                "sec_per_prompt": round(dt, 1),
                "peak_vram_gib": round(gib(torch.cuda.max_memory_allocated()), 2),
            }
            log.info("dim_batch=%d: %.1fs/prompt", db, dt)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            results["timings"][str(db)] = "OOM"
            log.info("dim_batch=%d: OOM", db)

    total = torch.cuda.get_device_properties(0).total_memory
    results["gpu"] = torch.cuda.get_device_name(0)
    results["gpu_total_gib"] = round(gib(total), 1)
    return results


def full_fit(model, dim_batch: int, n_prompts: int):
    prompts = jlens.examples.load_wikitext_prompts(n_prompts, min_chars=600)
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=dim_batch,
        max_seq_len=128,
        checkpoint_path=f"{OUT}/ckpt.pt",
        checkpoint_every=5,
        resume=True,
    )
    log.info("fitted: %r", lens)
    for l, J in lens.jacobians.items():
        assert J.shape == (3584, 3584), (l, J.shape)
        assert torch.isfinite(J).all(), f"non-finite values in J_{l}"
    log.info("finiteness check passed for %d layers", len(lens.jacobians))
    lens.save(f"{OUT}/qwen7b_lens.pt")
    log.info(
        "saved lens: %.2f GiB", gib(os.path.getsize(f"{OUT}/qwen7b_lens.pt"))
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--n-prompts", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    model, tok = load_model()

    if args.fit:
        full_fit(model, args.dim_batch, args.n_prompts)
    else:
        res = probe(model)
        with open(f"{OUT}/probe.json", "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
