"""GATE 3: render the layer x position slice page for prompt (a)."""

import pathlib

import torch
import transformers

import jlens
import jlens.vis

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "Fact: The currency used in the country shaped like a boot is"
OUT = pathlib.Path("out/qwen7b_slice.html")

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16
).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.load("out/qwen7b_lens.pt")

slice_data = jlens.vis.compute_slice(model, lens, PROMPT, mask_display=True)
html, raw, payload = jlens.vis.build_page(
    slice_data,
    PROMPT,
    title="Jacobian lens - Qwen2.5-7B-Instruct - boot/currency",
    description=(
        "J-lens slice: 100 WikiText prompts, all 27 source layers, dim_batch=64. "
        "Bottom row = model output."
    ),
    mode="embed",
)
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html) / 1e6:.1f} MB, payload {payload / 1e6:.1f} MB)")

# Mid-layer top-1 readout at the answer position (last token) for the report.
import json  # noqa: E402

ids = model.encode(PROMPT, max_length=512)[0].tolist()
toks = [tok.decode([t]) for t in ids]
mid_layers = [13, 14, 16, 19]
ll, _, _ = lens.apply(model, PROMPT, layers=mid_layers, positions=[-1, toks.index(" boot")])
report = {}
for i, pos_label in enumerate(("answer_pos_-1", "boot_pos")):
    report[pos_label] = {
        f"L{l}": tok.decode([ll[l][i].argmax().item()]) for l in mid_layers
    }
print(json.dumps(report, ensure_ascii=False, indent=2))
