# kokoro-lab-py

A small, faithful Python reproduction of broworkshop's **kokoro-lab** — built on
the upstream [`kokoro`](https://github.com/hexgrad/kokoro) pip package instead of
the bro/brosoundml C++ runtime.

Design a voice in Kokoro's style space, **watch the synthesis pipeline take shape
stage by stage**, and reshape its prosody (parametric VAD emotion) with a
re-decode of just the decoder back-half.

This is the Python leg of the plan to surface the kokoro-lab controls outside
bro; the in-browser (GitHub Pages) leg builds on the same shared math plus a
split-ONNX export.

## The idea: the lab is two layers, and only one needs the model

Everything kokoro-lab does splits cleanly, and this repo keeps the split:

**Layer A — pure NumPy over tiny artifacts (no model).** This is the bulk of the
"controls", and it's just linear algebra over four files that ship next to the
model in `brosoundml-data/kokoro/`:

| Artifact | Control | Math (`kokoro_lab/voicespace.py`) |
|---|---|---|
| `voice_basis.json` | the σ-axis **design sliders**, seeds, random | `style = mean + Σ coordsᵢ·stdᵢ·compsᵢ` |
| `emotion_basis.json` | **timbre** offset (per-emotion direction) | `style += Σ αₑ·fullₑ` |
| `masc_fem_basis.json` | the **masc↔fem** axis | `style += α·full.M` |
| `voice_bridge.bin` | **clone** an ECAPA embedding into the space | `style = ym + (x−xm)·B` |

Plus the **VAD emotion** panel (`kokoro_lab/emotion.py`) — pitch register/range,
energy, and rate as closed-form transforms of the predictor's own contours. No
model, no training.

**Layer B — the staged Kokoro forward (`kokoro_lab/engine.py`).** A step-by-step
reimplementation of `KModel.forward_with_tokens` that, unlike the stock one-shot
forward:

- captures every intermediate as a named **trace** stage
  (`bert_dur → d_en → t_en → pred_dur → F0_pred / N_pred → asr → audio`), and
- hands back a `DecodeContext` so the decoder back-half can be **re-decoded from
  edited prosody** (F0 / energy / duration) without re-running the front-end —
  i.e. brosoundml's `prepare_decode_context` + `decode_from`.

## Status

Validated against real Kokoro-82M weights (CUDA):

- ✅ Voice-space layer (offline): basis load, `coords↔style` round-trip,
  anchors, random, masc/fem + timbre offsets, clone bridge.
- ✅ Staged forward: all trace stages with correct shapes
  (e.g. 22 phonemes → `asr` 512×91, `F0_pred`/`N_pred` 182 = 2·91, 2.27 s audio).
- ✅ Decode-from-stage: editing energy re-decodes the back-half (same length,
  changed audio).
- ✅ Parametric emotion: arousal↑ retimes faster (fewer samples), arousal↓
  slower — retime + re-decode.
- ✅ Drawable control surfaces (in-browser, verified): paint F0 (pitch), N
  (energy), and pred_dur (per-phoneme timing); each stroke re-decodes just the
  back-half.
- ✅ Pinned prosody: a manual reshape is stored as a delta from the baseline and
  re-applied on every change, so it rides onto a different voice. Voice / VAD /
  masc-fem / speed all change the baseline and **compose** with the manual edit
  rather than resetting it.

## Setup

> **Python version matters.** `kokoro`'s g2p (misaki → spacy → thinc) has no
> wheels for Python 3.14 yet and won't compile there. Use **3.10–3.12**.

```bash
# with uv (recommended)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# point at your brosoundml-data/kokoro dir if it isn't auto-found
export KOKORO_LAB_DATA=/path/to/brosoundml-data/kokoro   # PowerShell: $env:KOKORO_LAB_DATA=...
```

The Kokoro-82M weights download from Hugging Face on first run. `torch` installs
the CPU build by default (plenty for an 82M model); for GPU, install a CUDA
`torch` build from the PyTorch index.

## Run

```bash
.venv/Scripts/python app.py          # Gradio UI
.venv/Scripts/python tests/smoke.py            # offline voice-space checks
.venv/Scripts/python tests/smoke.py --engine   # + load model, forward, decode-from
```

## Layout

```
kokoro_lab/
  voicespace.py   # Layer A: basis math + offsets + clone (NumPy, no model)
  emotion.py      # Layer A: VAD → prosody transforms
  engine.py       # Layer B: staged KModel forward + trace + decode-from
app.py            # Gradio UI: design sliders, drawable F0/N/dur surfaces,
                  #   VAD emotion, pinned-prosody composition
tests/smoke.py    # offline + engine smoke tests
```

## Not yet ported (vs the JS lab)

- `gen_in` / `har` trace stages (decoder-internal) — need a forward hook on the
  iSTFTNet decoder; the rest of the trace is captured.
- Clone is wired in `voicespace.style_from_ecapa`, but enrolling a clip needs an
  ECAPA speaker encoder (the JS lab uses the standalone ~18 MB artifact).
