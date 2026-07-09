"""GATE 2: apply the fitted lens locally (MPS) and check the known signature.

For each prompt, prints per-layer top-5 tokens at the requested positions,
J-lens (use_jacobian=True) vs logit-lens baseline (use_jacobian=False),
at layers spanning early/mid/late depth. Layers are reported both raw and
reindexed to [0, 100] (percent of depth, layer/(n_layers-1)*100).
"""

import argparse
import json
import time

import torch
import transformers

import jlens

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LENS_PATH = "out/qwen7b_lens.pt"

# Early/mid/late span of the 27 fitted source layers (0..26).
READ_LAYERS = [2, 6, 10, 13, 16, 19, 22, 24, 26]

PROMPT_A = "Fact: The currency used in the country shaped like a boot is"


def pct(layer: int, n_layers: int) -> int:
    return round(layer / (n_layers - 1) * 100)


def top5_table(model, tok, lens, prompt: str, positions: list[int]) -> None:
    print(f'\nPROMPT: "{prompt}"')
    ids = model.encode(prompt, max_length=512)
    toks = [tok.decode([t]) for t in ids[0].tolist()]
    for pos in positions:
        print(f"  position {pos} (token {toks[pos]!r}):")
        header = f"    {'layer':>5} {'%':>4} | {'J-lens top-5':<58} | logit-lens top-5"
        print(header)
        print("    " + "-" * (len(header) - 4))
        rows = {}
        for use_j in (True, False):
            ll, ml, _ = lens.apply(
                model, prompt, layers=READ_LAYERS, positions=[pos],
                max_seq_len=512, use_jacobian=use_j,
            )
            for layer in READ_LAYERS:
                tops = [
                    tok.decode([t]) for t in ll[layer][0].topk(5).indices.tolist()
                ]
                rows.setdefault(layer, {})[use_j] = tops
        for layer in READ_LAYERS:
            j5 = " ".join(repr(t) for t in rows[layer][True])
            b5 = " ".join(repr(t) for t in rows[layer][False])
            print(f"    {layer:>5} {pct(layer, model.n_layers):>4} | {j5:<58} | {b5}")
        # model's actual top-5 at this position
        _, ml, _ = lens.apply(model, prompt, layers=[READ_LAYERS[-1]],
                              positions=[pos], max_seq_len=512)
        actual = [tok.decode([t]) for t in ml[0].topk(5).indices.tolist()]
        print(f"    model actual next-token top-5: {actual}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--multihop-name", default="amazon-language")
    args = ap.parse_args()

    t0 = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).to("mps")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    print(f"model loaded in {time.perf_counter() - t0:.0f}s: {model!r}")

    lens = jlens.JacobianLens.load(LENS_PATH)
    print(f"lens: {lens!r}")

    top5_table(model, tok, lens, PROMPT_A, positions=[-1])

    with open("jacobian-lens/data/evaluations/lens-eval-multihop.json") as f:
        items = {it["name"]: it for it in json.load(f)["items"]}
    item = items[args.multihop_name]
    print(f"\nmultihop item {item['name']!r}: target={item['target']!r} "
          f"intermediates={item['intermediates']}")
    top5_table(model, tok, lens, item["prompt"], positions=[-1])
