"""Interactive slice explorer: type a prompt, get a J-lens slice page.

Loads Qwen2.5-7B-Instruct + the fitted lens once, then loops:

    prompt> Fact: The largest planet in the solar system is
    -> out/slices/fact-the-largest-planet....html  (opened in browser)

Commands:
    !gen N <prompt>   greedy-generate N tokens, then slice prompt+continuation.
                      The readout at each position depends only on its causal
                      prefix, so this is identical to a live per-token lens.
    !pin W [W2 ...]   force rank-tracking for these words in later pages, so
                      they are pinnable/charted even if never in any top-10
                      cell ("Italy" pins both "Italy" and " Italy" variants)
    !pin              list current pins;  !unpin  clears them
    !raw              toggle mask_display (word-like-only display tokens) off/on
    !stride N         render every Nth layer (default 1)
    q / quit          exit

Can also be used one-shot:  python scripts/slice_repl.py "your prompt here"
"""

import pathlib
import re
import subprocess
import sys
import time

import torch
import transformers

import jlens
import jlens.vis

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LENS_PATH = "out/qwen7b_lens.pt"
OUT_DIR = pathlib.Path("out/slices")


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48] or "prompt"


def resolve_pins(tok, words: list[str]) -> set[int]:
    """Single-token ids for each word, trying both bare and space-prefixed forms."""
    ids: set[int] = set()
    for word in words:
        for variant in (word, " " + word):
            t = tok.encode(variant, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
                print(f"  pinned {variant!r} (id {t[0]})")
            else:
                print(f"  {variant!r} tokenizes to {len(t)} tokens -> can't rank-track"
                      " (ranks are per vocab token); try a shorter form")
    return ids


def make_page(model, lens, prompt: str, *, mask_display: bool, stride: int,
              pins: set[int] | None = None) -> pathlib.Path:
    t0 = time.perf_counter()
    slice_data = jlens.vis.compute_slice(
        model, lens, prompt, mask_display=mask_display, layer_stride=stride,
        pinned_token_ids=pins or None,
    )
    html, _, _ = jlens.vis.build_page(
        slice_data,
        prompt,
        title=f"J-lens Qwen2.5-7B — {prompt[:60]}",
        description=f"lens: 100 prompts, 27 source layers. mask_display={mask_display}",
        mode="embed",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{slug(prompt)}.html"
    path.write_text(html, encoding="utf-8")
    print(f"  wrote {path} ({len(html) / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f}s")
    return path


def generate_text(model, hf, tok, prompt: str, n_new: int) -> str:
    """Greedy-decode n_new tokens; return the full text (prompt + continuation)."""
    ids = model.encode(prompt, max_length=512).to(hf.device)
    with torch.no_grad():
        out = hf.generate(
            ids,
            max_new_tokens=n_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    continuation = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"  continuation: {continuation!r}")
    full = tok.decode(out[0], skip_special_tokens=True)
    # The slice re-tokenizes the decoded text; warn if that changes the ids
    # (rare with BPE, but then the slice differs slightly from the live decode).
    re_ids = model.encode(full, max_length=1024)[0]
    if not torch.equal(re_ids.cpu(), out[0].cpu()):
        print("  note: retokenization differs from decode-time ids; "
              "cells near the seam may shift slightly")
    return full


def main() -> None:
    print("loading model + lens (one-time)...")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).to("mps")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(LENS_PATH)
    print(f"ready: {model!r} | {lens!r}\n")

    mask_display, stride = True, 1
    pins: set[int] = set()

    if len(sys.argv) > 1:  # one-shot mode (supports a leading "!gen N")
        text = " ".join(sys.argv[1:])
        if text.startswith("!gen"):
            _, n, text = text.split(maxsplit=2)
            text = generate_text(model, hf, tok, text, int(n))
        path = make_page(model, lens, text, mask_display=mask_display, stride=stride)
        subprocess.run(["open", str(path)])
        return

    while True:
        try:
            line = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("q", "quit", "exit"):
            break
        if line == "!raw":
            mask_display = not mask_display
            print(f"  mask_display -> {mask_display}")
            continue
        if line.startswith("!stride"):
            stride = max(1, int(line.split()[1]))
            print(f"  layer_stride -> {stride}")
            continue
        if line == "!unpin":
            pins.clear()
            print("  pins cleared")
            continue
        if line.startswith("!pin"):
            words = line.split()[1:]
            if words:
                pins |= resolve_pins(tok, words)
            else:
                print(f"  pins: {sorted(tok.decode([t]) for t in pins) or 'none'}")
            continue
        try:
            if line.startswith("!gen"):
                _, n, line = line.split(maxsplit=2)
                line = generate_text(model, hf, tok, line, int(n))
            path = make_page(model, lens, line, mask_display=mask_display,
                             stride=stride, pins=pins)
            subprocess.run(["open", str(path)])
        except Exception as e:  # keep the REPL alive on bad input
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
