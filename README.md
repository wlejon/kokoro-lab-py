# kokoro-lab-py

An interactive lab for the [Kokoro-82M](https://github.com/hexgrad/kokoro)
text-to-speech model. Design a voice in Kokoro's style space, **watch the
synthesis pipeline take shape stage by stage**, and **reshape its prosody by
drawing** on the pitch, energy, and timing contours — each stroke re-synthesizes
in place.

## What you can do

- **Design a voice** with sliders over the principal axes of Kokoro's style
  space (pitch, brightness, pace, …, plus character axes), seed from a stock
  voice, or roll a random one.
- **Nudge timbre** along a masc↔fem axis and per-emotion directions.
- **Steer emotion** with valence / arousal / dominance — a parametric transform
  of pitch register/range, energy, and speaking rate.
- **Draw the prosody**: paint the F0 (pitch), N (energy), and per-phoneme
  duration contours directly. Only the decoder back-half re-runs, so edits are
  fast.
- **Edits compose and persist**: a manual reshape is kept as a delta and
  re-applied as you change the voice, emotion, masc-fem, or speed — it rides
  along instead of resetting.

Everything re-renders on change; there is no render button.

## Setup

> **Use Python 3.10–3.12.** Kokoro's grapheme-to-phoneme stack (misaki → spacy)
> has no wheels for Python 3.13+ yet.

```bash
# create the environment (uv recommended; venv/pip works too)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

`torch` installs the CPU build by default — fine for an 82M model. For GPU,
install a CUDA `torch` build from the [PyTorch index](https://pytorch.org/get-started/locally/).

### Get the voice-space data

The slider basis and the timbre / masc-fem / clone artifacts live in the
[`wlejon/brosoundml-data`](https://huggingface.co/datasets/wlejon/brosoundml-data)
dataset on Hugging Face. Download the `kokoro/` folder and point the app at it:

```bash
huggingface-cli download wlejon/brosoundml-data --repo-type dataset \
    --include "kokoro/*" --local-dir ./data

export KOKORO_LAB_DATA=$PWD/data/kokoro     # PowerShell: $env:KOKORO_LAB_DATA = "$PWD/data/kokoro"
```

Only `voice_basis.json` is required; `emotion_basis.json`, `masc_fem_basis.json`,
and `voice_bridge.bin` are optional (their panels hide if absent). The Kokoro-82M
model weights themselves download automatically from Hugging Face on first run.

## Run

```bash
.venv/Scripts/python app.py                    # the Gradio lab
.venv/Scripts/python tests/smoke.py            # offline voice-space checks
.venv/Scripts/python tests/smoke.py --engine   # + load model, forward, decode-from
```

The app prints a local URL (default http://127.0.0.1:7860).

## How it works

The lab is two layers, and only one of them needs the model.

**Voice-space math — pure NumPy, no model** (`kokoro_lab/voicespace.py`,
`emotion.py`). Every control is linear algebra over a few small artifacts:

| Artifact | Control | Math |
|---|---|---|
| `voice_basis.json` | the σ-axis **design sliders**, seeds, random | `style = mean + Σ coordsᵢ·stdᵢ·compsᵢ` |
| `emotion_basis.json` | **timbre** offset (per-emotion direction) | `style += Σ αₑ·fullₑ` |
| `masc_fem_basis.json` | the **masc↔fem** axis | `style += α·full.M` |
| `voice_bridge.bin` | **clone** an ECAPA embedding into the space | `style = ym + (x−xm)·B` |

The VAD emotion panel is closed-form transforms of the predictor's own
pitch/energy/duration contours — no model, no training.

**The staged engine** (`kokoro_lab/engine.py`) runs Kokoro's forward pass step by
step so it can, unlike a one-shot call:

- capture every intermediate as a named **trace** stage
  (`bert_dur → d_en → t_en → pred_dur → F0_pred / N_pred → asr → audio`), and
- **re-decode just the back-half** from edited prosody (F0 / energy / duration)
  without re-running the front-end — this is what makes drawing on the contours
  cheap.

Edits are stored as a delta from the model's own prediction (a per-phoneme
duration ratio plus additive pitch/energy deltas), so they survive a change of
voice, emotion, masc-fem, or speed.

## Project layout

```
kokoro_lab/
  voicespace.py   # basis math + timbre/masc-fem offsets + clone (NumPy)
  emotion.py      # VAD → prosody transforms
  engine.py       # staged Kokoro forward + trace + decode-from-stage
app.py            # Gradio UI: design sliders, drawable F0/N/dur surfaces,
                  #   VAD emotion, composable pinned-prosody edits
tests/smoke.py    # offline + engine smoke tests
```

## Limitations

- The decoder-internal trace stages (`gen_in`, `har`) aren't surfaced yet; the
  rest of the pipeline trace is.
- Cloning a real clip into the style space is wired in
  (`voicespace.style_from_ecapa`) but needs an ECAPA speaker encoder to produce
  the embedding; the lab does not bundle one.

## License

MIT — see [LICENSE](LICENSE). This repository ships no model weights or voice
artifacts; those download from Hugging Face and carry their own licenses.
