"""kokoro-lab-py — Gradio UI.

A faithful, minimal reproduction of broworkshop's kokoro-lab: design a voice in
Kokoro's style space, watch the synthesis pipeline take shape, and reshape its
prosody by DRAWING on three control surfaces — F0 (pitch), N (energy), and
pred_dur (per-phoneme timing). Each stroke re-decodes just the decoder back-half.

Prosody edits are *pinned*: stored as a delta from the model's own prediction
(per-phoneme duration ratio + additive contour deltas) and re-applied on every
voice/text change, so a reshape rides along onto a different voice — exactly
kokoro-lab's retained-prosody model. Everything re-renders on change.

    python app.py

Set KOKORO_LAB_DATA to your brosoundml-data/kokoro dir if it isn't auto-found.
The Kokoro-82M weights download from Hugging Face on first run.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from kokoro_lab import VoiceSpace, KokoroLab
from kokoro_lab.emotion import Emotion

print("loading voice basis…")
VS = VoiceSpace.load()
print(f"  {VS.k} axes, {len(VS.names)} anchors, dim={VS.dim}")
print("loading Kokoro-82M (first run downloads from HF)…")
LAB = KokoroLab.load()
G2P = LAB.make_g2p("a")
print(f"  model on {LAB.device}")

SR = 24000


# ───────────────────────── static latent-stage plots ─────────────────────────
def _fig():
    fig, ax = plt.subplots(figsize=(7, 1.9), dpi=96)
    fig.patch.set_facecolor("#11141a"); ax.set_facecolor("#11141a")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#aaa", labelsize=7)
    return fig, ax


def plot_heat(m, title):
    fig, ax = _fig()
    m = np.asarray(m)
    if m.ndim == 1:
        m = m[None, :]
    ax.imshow(m, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    ax.set_title(f"{title} — {m.shape[0]}×{m.shape[1]}", color="#ddd", fontsize=9, loc="left")
    fig.tight_layout(); return fig


# ───────────────────────── prosody state (the "pin") ─────────────────────────
# State S carries: ctx (engine context), pred {F0,N,dur} (model prediction),
# cur {curF0,curN,curDur} (live, possibly edited), phon, and pin (the retained
# edit as a delta from pred). Ports kokoro-lab/lib/edit.js capture/reapply.
def _payload(S):
    L = len(S["curDur"]); phon = S["phon"]
    glyphs = [""] * L
    if len(phon) == L - 2:                       # boundary tokens bracket the phonemes
        glyphs = ["‹"] + list(phon) + ["›"]
    return json.dumps({
        "f0": [round(float(x), 3) for x in S["curF0"]],
        "n":  [round(float(x), 3) for x in S["curN"]],
        "dur": [int(x) for x in S["curDur"]],
        "glyphs": glyphs,
    })


def _meta(S, audio):
    return (f"{S.get('nph', 0)} phonemes · {int(np.sum(S['curDur']))} frames · "
            f"{len(audio) / SR:.2f}s · {LAB.device}"
            + ("  ·  ✎ prosody pinned" if S.get("pin") is not None else ""))


def _baseline(S):
    """The parametric baseline prosody for the current settings: the model
    prediction with the VAD emotion transform applied (rate → retime, then
    pitch/range/energy). Manual edits (the pin) layer ON TOP of this, so VAD /
    speed / masc-fem / voice all change the baseline without wiping the reshape.
    Returns (F0, N, dur) — all length-L in phonemes."""
    pred = S["pred"]
    f0 = pred["F0"].copy(); n = pred["N"].copy()
    dur = np.asarray(pred["dur"]).astype(int).copy()
    emo = Emotion(*S.get("emo", (0.0, 0.0, 0.0)))
    if emo.active():
        rate = emo.rate_scale()
        if abs(rate - 1.0) > 1e-3:
            ndur = np.maximum(1, np.round(dur / rate).astype(int))
            f0 = LAB.resample_by_dur(f0, dur, ndur)
            n = LAB.resample_by_dur(n, dur, ndur)
            dur = ndur
        f0, n = emo.transform_contours(f0, n)
    return f0, n, dur


def _capture_pin(S):
    """Remember the current on-screen prosody as a delta from the *baseline*
    (prediction + emotion), so re-deriving the baseline later doesn't double-count
    the emotion that was already visible when the user drew."""
    bF0, bN, bDur = _baseline(S); L = len(bDur)
    if len(S["curDur"]) != L:
        S["pin"] = None; return
    ratio = S["curDur"].astype(np.float64) / np.maximum(bDur, 1)
    f0_at_base = LAB.resample_by_dur(S["curF0"], S["curDur"], bDur)
    n_at_base = LAB.resample_by_dur(S["curN"], S["curDur"], bDur)
    S["pin"] = {"ratio": ratio, "dF0": f0_at_base - bF0,
                "dN": n_at_base - bN, "baseDur": bDur}


def _apply_pin(S):
    """Set cur* = baseline (prediction + emotion) with the retained manual reshape
    layered on. Drops the pin (→ cur = baseline) if the phoneme count no longer
    lines up (text changed)."""
    bF0, bN, bDur = _baseline(S); L = len(bDur)
    pin = S.get("pin")
    if not pin or len(pin["ratio"]) != L:
        S["pin"] = None
        S["curF0"] = bF0; S["curN"] = bN; S["curDur"] = bDur
        return
    target = np.maximum(1, np.round(bDur * pin["ratio"]).astype(int))
    d_f0 = LAB.resample_by_dur(pin["dF0"], pin["baseDur"], bDur)
    d_n = LAB.resample_by_dur(pin["dN"], pin["baseDur"], bDur)
    f0_ed = np.maximum(0, bF0 + d_f0); n_ed = bN + d_n
    S["curF0"] = LAB.resample_by_dur(f0_ed, bDur, target)
    S["curN"] = LAB.resample_by_dur(n_ed, bDur, target)
    S["curDur"] = target


# ───────────────────────── full synth (voice/text/emotion change) ─────────────────────────
def generate(state, text, mf, v, a, d, speed, *coords):
    coords = np.asarray(coords[:VS.k], np.float32)
    style = VS.add_mascfem(VS.style_from_coords(coords), float(mf))
    phon = LAB.phonemize(G2P, text)
    if not phon.strip():
        raise gr.Error("no phonemes for that text")
    audio, tr, ctx = LAB.synthesize(phon, style, speed=float(speed), trace=True)

    pred = {"F0": tr.stages["F0_pred"].astype(np.float32),
            "N":  tr.stages["N_pred"].astype(np.float32),
            "dur": np.asarray(tr.stages["pred_dur"]).astype(int)}
    prev_pin = state.get("pin") if isinstance(state, dict) else None
    S = {"ctx": ctx, "pred": pred, "phon": phon, "pin": prev_pin,
         "nph": len(tr.phoneme_ids), "emo": (float(v), float(a), float(d))}

    # cur = baseline (prediction + VAD emotion) + retained manual reshape. Any of
    # voice / masc-fem / speed / VAD changes the baseline; the manual edits ride on.
    _apply_pin(S)
    if S["pin"] is not None or Emotion(*S["emo"]).active():
        audio = LAB.decode_full(ctx, S["curDur"], S["curF0"], S["curN"])

    return ((SR, audio.astype(np.float32)), _meta(S, audio), phon, _payload(S),
            plot_heat(tr.stages["d_en"], "d_en"), plot_heat(tr.stages["asr"], "asr"), S)


# ───────────────────────── draw F0/N → decode-from (timing unchanged) ─────────────────────────
def decode_curve(edit_str, state):
    if not edit_str or not isinstance(state, dict):
        return gr.update(), gr.update(), state
    o = json.loads(edit_str); S = state
    S["curF0"] = np.asarray(o["f0"], np.float32)
    S["curN"] = np.asarray(o["n"], np.float32)
    audio = LAB.decode_full(S["ctx"], S["curDur"], S["curF0"], S["curN"])
    _capture_pin(S)
    return (SR, audio.astype(np.float32)), _meta(S, audio), S


# ───────────────────────── drag pred_dur → re-time (resample contours) ─────────────────────────
def decode_dur(edit_str, state):
    if not edit_str or not isinstance(state, dict):
        return gr.update(), gr.update(), gr.update(), state
    o = json.loads(edit_str); S = state
    new_dur = np.maximum(1, np.round(np.asarray(o["dur"], np.float64)).astype(int))
    if len(new_dur) != len(S["curDur"]):
        return gr.update(), gr.update(), gr.update(), state
    S["curF0"] = LAB.resample_by_dur(S["curF0"], S["curDur"], new_dur)
    S["curN"] = LAB.resample_by_dur(S["curN"], S["curDur"], new_dur)
    S["curDur"] = new_dur
    audio = LAB.decode_full(S["ctx"], new_dur, S["curF0"], S["curN"])
    _capture_pin(S)
    return (SR, audio.astype(np.float32)), _meta(S, audio), _payload(S), S  # redraw F0/N (lengths changed)


def clear_prosody(state):
    if isinstance(state, dict):
        state["pin"] = None
    return state


# ───────────────────────── slider seeding ─────────────────────────
def seed_coords(name):
    c = VS.anchor_coords(name)
    return [gr.update(value=float(c[i])) for i in range(VS.k)]


def random_coords():
    c = VS.random_coords()
    return [gr.update(value=float(c[i])) for i in range(VS.k)]


def neutral_coords():
    return [gr.update(value=0.0) for _ in range(VS.k)]


# ───────────────────────── the drawable-canvas frontend ─────────────────────────
# Self-bootstrapping: defines window.kokoroLab on first call, (re)attaches mouse
# listeners idempotently, then draws f0/n (curves) and dur (bars) from the
# payload. Ports kokoro-lab/lib/curves.js (paint) and the align row. Pushing an
# edit writes to a hidden bridge textbox → its .change fires the Python decode.
CURVE_JS = r"""
(s) => {
 if (!window.kokoroLab) {
  const PAD = 6;
  const L = {
    f0: [], n: [], dur: [], glyphs: [],
    rng: {f0: [0, 1], n: [0, 1], dur: 1},
    last: {f0: -1, n: -1, dur: -1}, lastV: {f0: 0, n: 0, dur: 0},
    range(arr) {
      let mn = Infinity, mx = -Infinity;
      for (const v of arr) { if (v < mn) mn = v; if (v > mx) mx = v; }
      mn = Math.min(0, mn); mx = mx * 1.35 + (mx <= 0 ? 1 : 0);
      if (mn === mx) { mn -= 1; mx += 1; }
      return [mn, mx];
    },
    drawCurve(which) {
      const arr = this[which], cv = document.getElementById('cv_' + which);
      if (!cv || !arr.length) return;
      const [mn, mx] = this.rng[which], ctx = cv.getContext('2d');
      const W = cv.width, H = cv.height, range = (mx - mn) || 1, n = arr.length;
      ctx.clearRect(0, 0, W, H);
      if (mn < 0 && mx > 0) {
        const zy = H - PAD - ((0 - mn) / range) * (H - 2 * PAD);
        ctx.strokeStyle = '#222b38'; ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(W, zy); ctx.stroke();
      }
      ctx.strokeStyle = which === 'f0' ? '#ffcf6b' : '#7fd1a6';
      ctx.lineWidth = 1.5; ctx.beginPath();
      for (let x = 0; x < W; x++) {
        const i = Math.floor(x * n / W);
        const y = H - PAD - ((arr[i] - mn) / range) * (H - 2 * PAD);
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    },
    drawDur() {
      const arr = this.dur, cv = document.getElementById('cv_dur');
      if (!cv || !arr.length) return;
      const ctx = cv.getContext('2d'), W = cv.width, H = cv.height, Ln = arr.length;
      let mx = 1; for (const vv of arr) if (vv > mx) mx = vv; mx *= 1.4;
      ctx.clearRect(0, 0, W, H);
      const bw = W / Ln;
      for (let i = 0; i < Ln; i++) {
        const h = (arr[i] / mx) * (H - 2 * PAD);
        ctx.fillStyle = '#6b9bff';
        ctx.fillRect(i * bw + 1, H - PAD - h, Math.max(1, bw - 2), h);
      }
      if (this.glyphs && this.glyphs.length === Ln && bw > 7) {
        ctx.fillStyle = '#8aa'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        for (let i = 0; i < Ln; i++) if (this.glyphs[i]) ctx.fillText(this.glyphs[i], i * bw + bw / 2, H - 1);
      }
    },
    draw(which) { which === 'dur' ? this.drawDur() : this.drawCurve(which); },
    setData(s) {
      const o = (typeof s === 'string') ? JSON.parse(s) : s;
      this.f0 = o.f0 || []; this.n = o.n || []; this.dur = o.dur || []; this.glyphs = o.glyphs || [];
      this.rng.f0 = this.range(this.f0); this.rng.n = this.range(this.n);
      this.draw('f0'); this.draw('n'); this.draw('dur');
    },
    paintCurve(which, e) {
      const cv = document.getElementById('cv_' + which), arr = this[which];
      if (!arr.length) return;
      const rect = cv.getBoundingClientRect(), H = cv.height;
      const [mn, mx] = this.rng[which], range = (mx - mn) || 1, n = arr.length;
      const xf = Math.max(0, Math.min(0.99999, (e.clientX - rect.left) / rect.width));
      const yPix = ((e.clientY - rect.top) / rect.height) * H;
      const i = Math.floor(xf * n);
      let v = Math.max(0, mn + ((H - PAD - yPix) / (H - 2 * PAD)) * range);
      this._fill(which, arr, i, v);
      this.draw(which);
    },
    paintDur(e) {
      const cv = document.getElementById('cv_dur'), arr = this.dur;
      if (!arr.length) return;
      const rect = cv.getBoundingClientRect(), H = cv.height, mx = this.rng.dur, Ln = arr.length;
      const xf = Math.max(0, Math.min(0.99999, (e.clientX - rect.left) / rect.width));
      const yPix = ((e.clientY - rect.top) / rect.height) * H;
      const i = Math.floor(xf * Ln);
      const v = Math.max(1, Math.round(((H - PAD - yPix) / (H - 2 * PAD)) * mx));
      this._fill('dur', arr, i, v, true);
      this.draw('dur');
    },
    _fill(which, arr, i, v, round) {
      const li = this.last[which], lv = this.lastV[which];
      if (li >= 0 && li !== i) {
        const a = Math.min(li, i), b = Math.max(li, i);
        const va = (li < i) ? lv : v, vb = (li < i) ? v : lv;
        for (let k = a; k <= b; k++) {
          let val = va + (vb - va) * ((b === a) ? 0 : (k - a) / (b - a));
          arr[k] = round ? Math.max(1, Math.round(val)) : val;
        }
      } else { arr[i] = v; }
      this.last[which] = i; this.lastV[which] = v;
    },
    paint(which, e) { which === 'dur' ? this.paintDur(e) : this.paintCurve(which, e); },
    push(which) {
      const id = which === 'dur' ? 'dur_out' : 'curve_out';
      const ta = document.querySelector('#' + id + ' textarea');
      if (!ta) return;
      ta.value = JSON.stringify(which === 'dur' ? {dur: this.dur} : {f0: this.f0, n: this.n});
      ta.dispatchEvent(new Event('input', {bubbles: true}));
    },
    attach(which) {
      const cv = document.getElementById('cv_' + which);
      if (!cv || cv._wired) return;
      cv._wired = true;
      let painting = false;
      cv.addEventListener('mousedown', (e) => {
        if (!this[which].length) return;
        e.preventDefault();
        if (which === 'dur') { let mx = 1; for (const vv of this.dur) if (vv > mx) mx = vv; this.rng.dur = mx * 1.4; }
        else this.rng[which] = this.range(this[which]);
        this.last[which] = -1; painting = true; this.paint(which, e);
      });
      window.addEventListener('mousemove', (e) => { if (painting) this.paint(which, e); });
      window.addEventListener('mouseup', () => { if (painting) { painting = false; this.push(which); } });
    },
  };
  window.kokoroLab = L;
 }
 window.kokoroLab.attach('f0'); window.kokoroLab.attach('n'); window.kokoroLab.attach('dur');
 window.kokoroLab.setData(s);
}
"""

CANVAS_HTML = """
<div style="display:flex;flex-direction:column;gap:10px">
  <div>
    <div style="color:#ffcf6b;font-size:13px;margin-bottom:2px">F0_pred — pitch (Hz) · drag to reshape</div>
    <canvas id="cv_f0" width="760" height="140"
            style="width:100%;background:#11141a;border:1px solid #333;border-radius:6px;cursor:crosshair"></canvas>
  </div>
  <div>
    <div style="color:#7fd1a6;font-size:13px;margin-bottom:2px">N_pred — energy · drag to reshape</div>
    <canvas id="cv_n" width="760" height="140"
            style="width:100%;background:#11141a;border:1px solid #333;border-radius:6px;cursor:crosshair"></canvas>
  </div>
  <div>
    <div style="color:#6b9bff;font-size:13px;margin-bottom:2px">pred_dur — frames per phoneme · drag to re-time</div>
    <canvas id="cv_dur" width="760" height="120"
            style="width:100%;background:#11141a;border:1px solid #333;border-radius:6px;cursor:crosshair"></canvas>
  </div>
</div>
"""

CSS = ".bridge{display:none !important}"


# ───────────────────────── UI ─────────────────────────
with gr.Blocks(title="kokoro-lab-py") as demo:
    gr.Markdown("## kokoro-lab-py — steer a voice through Kokoro's style space")
    st_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            text = gr.Textbox("Hello there. This is a test of the pipeline.",
                              label="text (enter to render)", lines=2)
            with gr.Row():
                seed = gr.Dropdown(["__neutral__"] + VS.names, value="__neutral__",
                                   label="seed voice")
                btn_rand = gr.Button("random", scale=0)
                btn_neu = gr.Button("neutral", scale=0)
                btn_clear = gr.Button("clear prosody", scale=0)

            sliders = []
            with gr.Accordion("voice design — σ axes", open=True):
                for i in range(VS.k):
                    lo, hi = float(VS.rng[i, 0] * 1.15), float(VS.rng[i, 1] * 1.15)
                    kind = "·attr" if VS.axis_kind[i] == "attr" else ""
                    sliders.append(gr.Slider(lo, hi, value=0.0, step=0.01,
                                             label=f"{VS.axis_name[i]}{kind}"))

            mf = gr.Slider(-3, 3, value=0.0, step=0.05, label="masc ↔ fem",
                           visible=VS.mascfem is not None)
            with gr.Accordion("emotion — prosody (VAD)", open=False):
                v = gr.Slider(-1, 1, value=0.0, step=0.05, label="valence")
                a = gr.Slider(-1, 1, value=0.0, step=0.05, label="arousal")
                d = gr.Slider(-1, 1, value=0.0, step=0.05, label="dominance")
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="speed")

        with gr.Column(scale=1):
            audio = gr.Audio(label="audio", autoplay=True)
            meta = gr.Markdown()
            phon = gr.Textbox(label="phonemes", interactive=False)
            gr.HTML(CANVAS_HTML)
            curve_out = gr.Textbox(elem_id="curve_out", elem_classes=["bridge"])  # F0/N edit → Python
            dur_out = gr.Textbox(elem_id="dur_out", elem_classes=["bridge"])      # dur edit → Python
            curve_in = gr.Textbox(elem_classes=["bridge"])                        # Python → JS (draw)
            with gr.Accordion("latent stages", open=False):
                p_den = gr.Plot(label="d_en")
                p_asr = gr.Plot(label="asr")

    inputs_all = [st_state, text, mf, v, a, d, speed, *sliders]
    gen_out = [audio, meta, phon, curve_in, p_den, p_asr, st_state]

    def _draw(ev):
        """Chain: after Python updates curve_in, (re)draw the canvases via CURVE_JS."""
        return ev.then(fn=None, inputs=[curve_in], outputs=None, js=CURVE_JS)

    # everything on change → full re-synth (retained prosody rides along)
    _draw(text.submit(generate, inputs_all, gen_out))
    for s in [*sliders, mf, v, a, d, speed]:
        _draw(s.release(generate, inputs_all, gen_out))
    _draw(seed.change(seed_coords, seed, sliders).then(generate, inputs_all, gen_out))
    _draw(btn_rand.click(random_coords, None, sliders).then(generate, inputs_all, gen_out))
    _draw(btn_neu.click(neutral_coords, None, sliders).then(generate, inputs_all, gen_out))
    _draw(btn_clear.click(clear_prosody, st_state, st_state).then(generate, inputs_all, gen_out))

    # drawing a contour / re-timing → re-decode just the back half
    curve_out.change(decode_curve, [curve_out, st_state], [audio, meta, st_state])
    _draw(dur_out.change(decode_dur, [dur_out, st_state], [audio, meta, curve_in, st_state]))

    _draw(demo.load(generate, inputs_all, gen_out))


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=CSS)
