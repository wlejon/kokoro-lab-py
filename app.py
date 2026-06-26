"""kokoro-lab-py — Gradio UI.

A faithful, minimal reproduction of broworkshop's kokoro-lab: design a voice in
Kokoro's style space, watch the synthesis pipeline take shape stage by stage,
and reshape its prosody (parametric VAD emotion) with a re-decode of just the
back half.

    python app.py            # loads the basis + KModel, serves the UI

Set KOKORO_LAB_DATA to your brosoundml-data/kokoro dir if it isn't auto-found.
The Kokoro-82M weights download from Hugging Face on first run.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from kokoro_lab import VoiceSpace, KokoroLab
from kokoro_lab.emotion import Emotion

# ───────────────────────── load once ─────────────────────────
print("loading voice basis…")
VS = VoiceSpace.load()
print(f"  {VS.k} axes, {len(VS.names)} anchors, dim={VS.dim}")
print("loading Kokoro-82M (first run downloads from HF)…")
LAB = KokoroLab.load()
G2P = LAB.make_g2p("a")
print(f"  model on {LAB.device}")

SR = 24000


# ───────────────────────── plotting ─────────────────────────
def _fig():
    fig, ax = plt.subplots(figsize=(7, 2.2), dpi=96)
    fig.patch.set_facecolor("#11141a")
    ax.set_facecolor("#11141a")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#aaa", labelsize=7)
    return fig, ax


def plot_curve(y, title, color):
    fig, ax = _fig()
    ax.plot(np.asarray(y).ravel(), color=color, lw=1.0)
    ax.set_title(title, color="#ddd", fontsize=9, loc="left")
    fig.tight_layout()
    return fig


def plot_dur(dur, ids):
    fig, ax = _fig()
    dur = np.asarray(dur).ravel()
    ax.bar(np.arange(len(dur)), dur, color="#6b9bff", width=0.9)
    ax.set_title(f"pred_dur — {len(dur)} tokens, {int(dur.sum())} frames",
                 color="#ddd", fontsize=9, loc="left")
    fig.tight_layout()
    return fig


def plot_heat(m, title):
    fig, ax = _fig()
    m = np.asarray(m)
    if m.ndim == 1:
        m = m[None, :]
    ax.imshow(m, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    ax.set_title(f"{title} — {m.shape[0]}×{m.shape[1]}", color="#ddd",
                 fontsize=9, loc="left")
    fig.tight_layout()
    return fig


# ───────────────────────── core action ─────────────────────────
def generate(text, mf, v, a, d, speed, *coords):
    coords = np.asarray(coords[:VS.k], np.float32)
    style = VS.style_from_coords(coords)
    style = VS.add_mascfem(style, float(mf))

    phon = LAB.phonemize(G2P, text)
    if not phon.strip():
        raise gr.Error("no phonemes for that text")

    audio, tr, ctx = LAB.synthesize(phon, style, speed=float(speed), trace=True)

    emo = Emotion(float(v), float(a), float(d))
    f0, n = tr.stages["F0_pred"], tr.stages["N_pred"]
    dur = tr.stages["pred_dur"]
    if emo.active():
        audio, etr = LAB.apply_emotion(ctx, emo)
        f0, n, dur = etr.stages["F0_pred"], etr.stages["N_pred"], etr.stages["pred_dur"]

    meta = (f"{len(tr.phoneme_ids)} phonemes · {int(np.sum(dur))} frames · "
            f"{len(audio) / SR:.2f}s audio")
    return (
        (SR, audio.astype(np.float32)),
        phon, meta,
        plot_curve(f0, "F0_pred — pitch (Hz, frame rate)", "#ffcf6b"),
        plot_curve(n, "N_pred — energy (frame rate)", "#7fd1a6"),
        plot_dur(dur, tr.phoneme_ids),
        plot_heat(tr.stages["d_en"], "d_en — predictor conditioning"),
        plot_heat(tr.stages["asr"], "asr — duration-aligned content"),
    )


def seed_coords(name):
    c = VS.anchor_coords(name)
    return [gr.update(value=float(c[i])) for i in range(VS.k)]


def random_coords():
    c = VS.random_coords()
    return [gr.update(value=float(c[i])) for i in range(VS.k)]


def neutral_coords():
    return [gr.update(value=0.0) for _ in range(VS.k)]


# ───────────────────────── UI ─────────────────────────
with gr.Blocks(title="kokoro-lab-py", theme=gr.themes.Base()) as demo:
    gr.Markdown("## kokoro-lab-py — steer a voice through Kokoro's style space")

    with gr.Row():
        with gr.Column(scale=1):
            text = gr.Textbox("Hello there. This is a test of the pipeline.",
                              label="text", lines=2)
            with gr.Row():
                seed = gr.Dropdown(["__neutral__"] + VS.names, value="__neutral__",
                                   label="seed voice")
                btn_rand = gr.Button("random", scale=0)
                btn_neu = gr.Button("neutral", scale=0)

            sliders = []
            with gr.Accordion("voice design — σ axes", open=True):
                for i in range(VS.k):
                    lo, hi = float(VS.rng[i, 0] * 1.15), float(VS.rng[i, 1] * 1.15)
                    kind = "·attr" if VS.axis_kind[i] == "attr" else ""
                    sliders.append(gr.Slider(
                        lo, hi, value=0.0, step=0.01,
                        label=f"{VS.axis_name[i]}{kind}"))

            mf = gr.Slider(-3, 3, value=0.0, step=0.05,
                           label="masc ↔ fem",
                           visible=VS.mascfem is not None)

            with gr.Accordion("emotion — prosody (VAD)", open=False):
                v = gr.Slider(-1, 1, value=0.0, step=0.05, label="valence")
                a = gr.Slider(-1, 1, value=0.0, step=0.05, label="arousal")
                d = gr.Slider(-1, 1, value=0.0, step=0.05, label="dominance")
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="speed")

            btn_run = gr.Button("generate", variant="primary")

        with gr.Column(scale=1):
            audio = gr.Audio(label="audio", autoplay=True)
            meta = gr.Markdown()
            phon = gr.Textbox(label="phonemes", interactive=False)
            p_f0 = gr.Plot(label="F0_pred")
            p_n = gr.Plot(label="N_pred")
            p_dur = gr.Plot(label="pred_dur")
            p_den = gr.Plot(label="d_en")
            p_asr = gr.Plot(label="asr")

    outputs = [audio, phon, meta, p_f0, p_n, p_dur, p_den, p_asr]
    inputs = [text, mf, v, a, d, speed, *sliders]

    btn_run.click(generate, inputs=inputs, outputs=outputs)
    seed.change(seed_coords, inputs=seed, outputs=sliders)
    btn_rand.click(random_coords, outputs=sliders)
    btn_neu.click(neutral_coords, outputs=sliders)


if __name__ == "__main__":
    demo.launch()
