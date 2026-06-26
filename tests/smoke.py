"""Smoke tests. Run:  python tests/smoke.py

The voice-space test is offline (no model, no network) and validates the basis
math + artifact loading. The engine test loads Kokoro-82M (downloads on first
run) and exercises the staged forward + a decode-from re-decode.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kokoro_lab import VoiceSpace


def test_voicespace():
    vs = VoiceSpace.load()
    assert vs.dim == 256 and vs.k > 0
    print(f"[ok] basis: k={vs.k} dim={vs.dim} anchors={len(vs.names)} "
          f"mascfem={'y' if vs.mascfem is not None else 'n'} "
          f"timbre={len(vs.emotions)} bridge={'y' if vs.bridge else 'n'}")

    # neutral coords -> mean style
    style0 = vs.style_from_coords(np.zeros(vs.k, np.float32))
    assert np.allclose(style0, vs.mean, atol=1e-5)
    print("[ok] neutral coords reconstruct the mean style")

    # round-trip: coords -> style -> coords (within the basis span)
    c = vs.random_coords(np.random.default_rng(0))
    style = vs.style_from_coords(c)
    c2 = vs.coords_from_style(style)
    err = float(np.max(np.abs(c - c2)))
    assert err < 1e-3, err
    print(f"[ok] coords->style->coords round-trip (max err {err:.2e})")

    # an anchor seeds a real voice
    if vs.names:
        a = vs.anchor_coords(vs.names[0])
        assert a.shape == (vs.k,)
        print(f"[ok] anchor '{vs.names[0]}' -> coords")


def test_engine():
    from kokoro_lab import KokoroLab
    from kokoro_lab.emotion import Emotion
    vs = VoiceSpace.load()
    lab = KokoroLab.load()
    g2p = lab.make_g2p("a")
    style = vs.style_from_coords(vs.anchor_coords(
        "af_heart" if "af_heart" in vs.names else vs.names[0]))

    phon = lab.phonemize(g2p, "Hello there.")
    audio, tr, ctx = lab.synthesize(phon, style, trace=True)
    assert audio.ndim == 1 and audio.size > 0
    for k in ("bert_dur", "d_en", "t_en", "pred_dur", "F0_pred", "N_pred", "asr"):
        assert k in tr.stages, k
    print(f"[ok] forward: {len(tr.phoneme_ids)} phonemes, {audio.size} samples, "
          f"stages={list(tr.stages)}")

    # decode-from: re-decode with energy halved — must change audio, same length
    audio2 = lab.decode(ctx, n=ctx.n * 0.5)
    assert audio2.shape == audio.shape
    assert not np.allclose(audio2, audio)
    print("[ok] decode-from re-decodes the back half (edited energy)")

    # parametric emotion (excited): faster -> fewer samples
    a3, _ = lab.apply_emotion(ctx, Emotion(a=0.8))
    print(f"[ok] emotion re-decode: {audio.size} -> {a3.size} samples (arousal+)")


if __name__ == "__main__":
    test_voicespace()
    if "--engine" in sys.argv:
        test_engine()
    else:
        print("(skipping engine test; pass --engine to load Kokoro-82M)")
