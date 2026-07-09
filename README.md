# Jacobian Lens on Qwen2.5-7B-Instruct

Fitting and applying the **Jacobian lens** (J-lens) from Anthropic's
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
to `Qwen/Qwen2.5-7B-Instruct`, using the reference implementation
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (vendored in
[jacobian-lens/](jacobian-lens/)).

## What the J-lens is

A **logit lens** decodes an intermediate residual-stream activation `h_l` (layer `l`,
some position) directly through the model's final norm + unembedding, pretending the
final 28−l layers don't exist. It works poorly in early/mid layers because the
residual stream's *basis drifts* across depth.

The **Jacobian lens** fixes the basis mismatch with a learned linear transport: it
multiplies `h_l` by the **corpus-averaged input–output Jacobian** before unembedding:

```
lens_l(h) = unembed( J_l @ h ),    J_l = E[ ∂h_final / ∂h_l ]
```

`J_l` is a `[d_model, d_model]` matrix per source layer — the average linearization of
"everything the remaining layers do" to a perturbation at layer `l`. The expectation is
over prompts and positions in a generic web-text corpus (the estimator injects a one-hot
cotangent at every valid *target* position at once, backprops, and averages the
resulting gradient over *source* positions; see the `jlens/fitting.py` module docstring).

## What implementing it requires

### 1. Environment (done)

- The package requires `transformers>=5.5`, which can conflict with pinned tooling →
  **fresh venv** at [.venv/](.venv/), untouched system Python.
- Installed: `torch 2.13.0` (MPS backend for Apple Silicon), `transformers 5.13.0`,
  `jlens` (editable from the cloned repo), `datasets` (for the WikiText fitting corpus).
- Verify: `import jlens` works; `jlens.from_hf(hf_model, tok)` auto-detects the
  Qwen layout (residual blocks at `model.layers`, final norm `model.norm`, unembed
  `lm_head`).

### 2. Fitting — the expensive part

Per prompt, `jlens.fit` runs **one forward pass** (prompt replicated `dim_batch`×
along the batch axis, graph retained) and **`ceil(d_model / dim_batch)` backward
passes**, each recovering `dim_batch` rows of every `J_l` simultaneously.

Key numbers for Qwen2.5-7B-Instruct:

| quantity | value |
|---|---|
| `n_layers` / `d_model` | 28 / 3584 |
| backward passes/prompt @ `dim_batch=8` | ⌈3584/8⌉ = **448** |
| backward passes/prompt @ `dim_batch=64` | 56 (same total FLOPs, better GPU utilization) |
| checkpoint / lens size (all 27 source layers) | 27 × 3584² × 4 B ≈ **1.39 GB** (fp32 ckpt), half that for the fp16 saved lens |
| model weights (bf16) | ≈ 15.2 GB |

Levers:
- `dim_batch` — the speed knob. Total backward FLOPs are constant; larger values mean
  fewer, fatter passes. Memory cost: activations for a `dim_batch × 128`-token forward
  with retained graph across all source layers.
- `source_layers` — does **not** change backward count, only per-pass graph depth and
  checkpoint size.
- `checkpoint_path` + `resume=True` — running Jacobian sum written atomically every
  `checkpoint_every` prompts, so interrupts are recoverable mid-fit.
- Corpus: `jlens.examples.load_wikitext_prompts(n)`; the paper uses 1000 × 128-token
  sequences but quality saturates fast (~100 prompts usable). The first 16 positions
  are excluded (attention sinks — atypical residual statistics).

### 3. Compute reality check on this machine

This host is an Apple M5 with **24 GB unified memory**.

**Measured** — 0.5B smoke test (Qwen2.5-0.5B-Instruct, MPS, `dim_batch=8`, all 23
source layers, `max_seq_len=128`): **25.5 s/prompt**, 2.85 GiB peak driver memory.
Pipeline validated end-to-end (fit → apply) on Apple Silicon.

**Extrapolated to 7B on this Mac**: per-prompt FLOPs scale ≈ 56× (4× more backward
passes × ~14× per-pass cost) → **~20–25 min/prompt**, i.e. ~40 hours for 100 prompts.
Memory: 15.2 GB weights + a retained graph that scales to ~9 GB ≈ 24 GB, at/over the
machine's total unified memory. **Fitting the 7B locally is infeasible; fitting is done
on a rented cloud GPU. Applying the fitted lens (forward passes only, no graph) is
feasible locally.**

### 4. Apply & evaluate

`lens.apply(model, prompt, positions=[-1])` returns per-layer lens logits at chosen
positions — decoded top-k tokens per layer, next to a `use_jacobian=False` logit-lens
baseline. Expected qualitative signature on e.g.
*"Fact: The currency used in the country shaped like a boot is"*:
mid layers surface the unspoken intermediate concept (**Italy → lira/euro**), late
layers converge to the model's actual next token, and the logit-lens baseline is
degraded at early/mid depth relative to the J-lens.

### 5. Visualization

`jlens.vis.compute_slice(model, lens, prompt)` + `build_page(mode="embed")` renders a
self-contained layer × position HTML page: each cell is the lens top-1 token at that
(position, layer); clicking pins a token and shows rank-tracking charts.

### 6. Remote fitting workflow (RunPod)

The fit runs on a rented RunPod GPU (H100 80GB preferred, A100-80GB fallback —
the binding constraint is VRAM: 15.2 GB weights + a retained graph of ~9 GB at
`dim_batch=8`, ~36 GB at `dim_batch=32`). 60 GB container disk (model cache +
checkpoint + lens); no network volume — in-pod checkpointing covers interrupts and
the lens is downloaded before the pod is terminated.

- [scripts/runpod_driver.py](scripts/runpod_driver.py) — local: `create` / `status` /
  `terminate` a pod via the RunPod REST API. Reads the key from `~/.runpod/api_key`;
  injects a dedicated SSH keypair (`out/runpod_ssh_key`) via the image's `PUBLIC_KEY` env.
- [scripts/remote_probe_and_fit.py](scripts/remote_probe_and_fit.py) — runs on the pod:
  - *probe* (default): 2-prompt fit at `dim_batch=8` + 1-prompt timings at 32/64 →
    `out/probe.json` (sec/prompt, peak VRAM). GATE 0 numbers on real fit hardware.
  - *fit* (`--fit --dim-batch N --n-prompts M`): full fit, `checkpoint_every=5`,
    `resume=True`, finiteness + shape asserts, saves `out/qwen7b_lens.pt` (fp16).
- The fitted lens (~0.7 GB) is `scp`'d back to the Mac; GATE 2/3 (apply + slice
  visualization) run locally on MPS — apply is forward-only, so the M5 handles it.

**Measured on the pod (H100 80GB, `runpod/pytorch` image)** — 7B, all 27 source
layers, seq 128:

| `dim_batch` | sec/prompt | peak VRAM | 100 prompts |
|---|---|---|---|
| 8 | 21.8 | 18.9 GiB | ~36 min |
| 32 | 19.2 | 32.6 GiB | ~32 min |
| **64 (chosen)** | **18.1** | **50.9 GiB** | **~30 min** |

Backward count barely matters on H100 — graph replay is cheap vs kernel throughput.
Chosen fit: **100 WikiText prompts, all 27 source layers, `dim_batch=64`**.

Pod-environment gotchas hit (and fixes):
1. `runpod/pytorch` ships torch 2.4.1 — too old for `transformers>=5.5` → upgraded to
   torch 2.6.0+cu124.
2. The image's stale **torchvision 0.19.1** then breaks *every* transformers model
   import (`operator torchvision::nms does not exist` surfacing as
   `Could not import module 'Qwen2ForCausalLM'`) → `pip uninstall torchvision
   torchaudio` (transformers only imports torchvision if present).
3. `device_map=` requires **`accelerate`** → installed.

### 7. Results (GATE 2): the lens reproduces the paper's signature

Applied locally on the M5 (MPS, forward-only). Layers reported as % of depth
(`layer / 27 × 100`). Highlights, `use_jacobian=True`:

*"Fact: The currency used in the country shaped like a boot is"* — at the **' boot'**
token position, mid-band layers surface the **unspoken intermediate**:

| depth | J-lens top-5 at ' boot' |
|---|---|
| 59% (L16) | `——` **`意大利`** `—` **`Italian`** **`Italy`** |
| 70% (L19) | **`意大利` `Italian` `Italy` ` Italian` ` Italy`** |
| 89% (L24) | `形状` `-shaped` ` shaped` ` is` ` shape` |

"Italy" never appears in the prompt. At the final position (' is'), 89% depth reads
`欧元 / euros / currency / euro` and 96% converges to ` euros` — the model's actual
next-token top-5 is `[' the', ' euros', ' called', ...]`.

Multi-hop (`amazon-language`: "...language ... where the Amazon River ends is",
target *Portuguese*, intermediate *Brazil*): at the answer position, 81% depth reads
language-concept tokens (`英语/English/语言`), **89% reads
`Portuguese / Brazilian / Spanish / Brazil`** (the intermediate visible), 96%
converges to ` Portuguese` = the model's top next token.

The **logit-lens baseline** (`use_jacobian=False`) at the same layers is garbage below
~80% depth (`'bellion'`, `' retal'`, `'icester'`…) and only partially recovers late —
exactly the degradation the J-lens transport is meant to fix. One honest caveat: at
*non-concept-bearing* positions the J-lens mid-band mostly reads whitespace/punctuation
(a "nothing verbalizable here" readout); the semantic content is position-localized,
which is what the GATE 3 slice view shows.

## Progress log

- **GATE 0** — env setup + probes ✅ (0.5B MPS smoke test; 7B H100 probe: 18–22 s/prompt)
- **GATE 1** — fit ✅ (100 WikiText prompts, all 27 source layers, `dim_batch=64`,
  ~34 min H100; all Jacobians finite, `[3584,3584]`; lens = [out/qwen7b_lens.pt](out/qwen7b_lens.pt), 0.65 GiB)
- **GATE 2** — apply + signature check ✅ (see §7)
- **GATE 3** — slice visualization ✅ — [out/qwen7b_slice.html](out/qwen7b_slice.html)
  (1.8 MB self-contained page, d3 inlined; `compute_slice` + `build_page(mode="embed")`,
  `mask_display=True`). Mid-layer top-1 at the ' boot' position: `——`/`—` at L13–L16,
  **`意大利` (Italy) at L19**; at the answer position mid layers read whitespace/filler
  (top-1 `stdarg`/newline — no verbalizable content parked there until ~L24, where
  `欧元`/euro takes over; see §7).

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -e ./jacobian-lens datasets
# GATE 0 smoke test (local, ~1 min after download):
.venv/bin/python scripts/probe_smoke_05b.py
# GATE 0/1 on a rented GPU (80GB-class; see §6 pod gotchas):
scp scripts/remote_probe_and_fit.py <pod>: && ssh <pod> python3 remote_probe_and_fit.py           # probe
ssh <pod> python3 remote_probe_and_fit.py --fit --dim-batch 64 --n-prompts 100                    # fit
scp <pod>:out/qwen7b_lens.pt out/
# ...or skip fitting entirely — the fitted lens is published on the Hub:
.venv/bin/python -c "from huggingface_hub import hf_hub_download; import shutil; \
  shutil.copy(hf_hub_download('dormantx/jacobian-lens-qwen2.5-7b-instruct', 'qwen7b_lens.pt'), 'out/')"
# GATE 2 + 3 (local, MPS):
.venv/bin/python scripts/apply_lens.py
.venv/bin/python scripts/make_slice.py
```

## Exploring: `scripts/slice_repl.py`

Interactive explorer — loads model + lens once, then each typed prompt becomes a slice
page in `out/slices/` (~20–60 s each on the M5):

```
prompt> Fact: The capital of the country whose flag has a red maple leaf is
prompt> !gen 30 Q: Which is heavier, a kg of feathers or a kg of steel? A:
prompt> !pin lira Rome     # force rank-tracking for words never in any top-10 cell
prompt> !raw / !stride 2 / q
```

`!gen N` greedy-decodes N tokens and slices *prompt + continuation* — because each
position's readout depends only on its causal prefix, this is **identical to a live
per-token lens during generation**; streaming would change the UX, not the numbers.
Note the J-lens needs residual-stream activations, so it can never run against a
token-only API (OpenRouter etc.) — the model must be held in-process (locally here,
or the same scripts on a rented GPU for speed). Long generations: use `!stride 2` —
embedded page size grows with positions × tracked tokens.

`!pin <word> [...]` makes a token chartable even if it never appears in any cell's
top-10 (the page only stores rank data for tracked tokens). Pins apply to pages
generated *after* the command and persist until `!unpin`. Constraint: ranks are per
vocab token, so only single-token words work — `' Rome'` is one Qwen token (id 21718)
and pins fine, but `' lira'` splits into two BPE tokens and can't be rank-tracked
(which is also why "lira" could never appear as a top-5 token in the GATE 2 tables).

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
