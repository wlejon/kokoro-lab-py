"""Voice-space math — a NumPy port of kokoro-lab's designer.js / clone.js /
timbre.js / mascfem.js. Pure linear algebra over the static basis artifacts;
no model is involved.

The single currency is a 256-D Kokoro style vector (`ref_s`): the first 128
dims drive the decoder, the last 128 drive the prosody predictor. Every control
in the lab is a way of producing or nudging that vector:

    style = mean + Σ_i coords_i · std_i · comps_i        (the slider designer)
    style += Σ_e α_e · emotion_full_e                     (timbre offset)
    style += α · mascfem_full_M                           (masc↔fem offset)
    style = ym + (x − xm) · B                             (ECAPA clone bridge)
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _find_default_root() -> Optional[str]:
    """Resolve a brosoundml-data/kokoro dir the way the JS lab does: an env
    override, then the usual sibling-repo spot."""
    env = os.environ.get("KOKORO_LAB_DATA")
    candidates = [
        env,
        env and os.path.join(env, "kokoro"),
        os.path.expanduser("~/projects/brosoundml-data/kokoro"),
        "D:/projects/brosoundml-data/kokoro",
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "voice_basis.json")):
            return c
    return None


@dataclass
class VoiceSpace:
    # --- voice_basis.json (the slider designer) ---
    dim: int
    k: int
    mean: np.ndarray                  # (dim,)
    comps: np.ndarray                 # (k, dim)
    std: np.ndarray                   # (k,)
    rng: np.ndarray                   # (k, 2) — [lo, hi] per axis, in σ units
    anchors: np.ndarray               # (n_named, k) — coords of the stock voices
    names: list[str]                  # anchor names (af_heart, ...)
    axis_name: list[str]              # per-axis label
    axis_kind: list[str]              # 'attr' | 'char'
    var_explained: np.ndarray         # (k,)
    # --- optional offset bases ---
    emotions: list[str] = field(default_factory=list)      # ANG/DIS/...
    emotion_full: dict[str, np.ndarray] = field(default_factory=dict)
    mascfem: Optional[np.ndarray] = None                   # (dim,) — toward M pole
    # --- optional clone bridge (voice_bridge.bin) ---
    bridge: Optional[dict] = None     # {D, M, xm, ym, B}

    # ───────────────────────── loading ─────────────────────────
    @classmethod
    def load(cls, root: Optional[str] = None) -> "VoiceSpace":
        root = root or _find_default_root()
        if not root:
            raise FileNotFoundError(
                "voice_basis.json not found. Set KOKORO_LAB_DATA to your "
                "brosoundml-data/kokoro dir."
            )
        with open(os.path.join(root, "voice_basis.json"), "r", encoding="utf-8") as f:
            b = json.load(f)
        k = b["k"]
        vs = cls(
            dim=b["dim"], k=k,
            mean=np.asarray(b["mean"], np.float32),
            comps=np.asarray(b["comps"], np.float32),
            std=np.asarray(b["std"], np.float32),
            rng=np.asarray(b["range"], np.float32),
            anchors=np.asarray(b["anchors"], np.float32),
            names=list(b["names"]),
            axis_name=list(b.get("axisName", [f"PC{i+1}" for i in range(k)])),
            axis_kind=list(b.get("axisKind", ["char"] * k)),
            var_explained=np.asarray(b.get("varExplained", [0.0] * k), np.float32),
        )
        vs._load_optional(root)
        return vs

    def _load_optional(self, root: str) -> None:
        # timbre directions (emotion_basis.json) — panel is optional
        p = os.path.join(root, "emotion_basis.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                eb = json.load(f)
            full = eb.get("full") or {}
            self.emotions = list(eb.get("emotions", []))
            self.emotion_full = {
                e: np.asarray(full[e], np.float32) for e in self.emotions if e in full
            }
        # masc↔fem axis (masc_fem_basis.json) — optional
        p = os.path.join(root, "masc_fem_basis.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                mf = json.load(f)
            if mf.get("full", {}).get("M") is not None:
                self.mascfem = np.asarray(mf["full"]["M"], np.float32)
        # ECAPA→style clone bridge (voice_bridge.bin) — optional
        p = os.path.join(root, "voice_bridge.bin")
        if os.path.exists(p):
            with open(p, "rb") as f:
                buf = f.read()
            D, M = struct.unpack_from("<ii", buf, 0)
            off = 8
            xm = np.frombuffer(buf, np.float32, D, off); off += 4 * D
            ym = np.frombuffer(buf, np.float32, M, off); off += 4 * M
            B = np.frombuffer(buf, np.float32, D * M, off).reshape(D, M)
            self.bridge = {"D": D, "M": M, "xm": xm, "ym": ym, "B": B}

    # ───────────────────────── designer ─────────────────────────
    def style_from_coords(self, coords: np.ndarray) -> np.ndarray:
        """coords (σ units, length k) → 256-D style vector."""
        coords = np.asarray(coords, np.float32)
        return self.mean + (coords * self.std) @ self.comps

    def coords_from_style(self, style: np.ndarray) -> np.ndarray:
        """256-D style → coords (σ units) — the projection used by clone."""
        style = np.asarray(style, np.float32)
        std = np.where(self.std == 0, 1.0, self.std)
        return (self.comps @ (style - self.mean)) / std

    def anchor_coords(self, name: str) -> np.ndarray:
        """Seed coords from a named stock voice, or the neutral centroid."""
        if name in ("__neutral__", "", None):
            return np.zeros(self.k, np.float32)
        i = self.names.index(name)
        return self.anchors[i].copy()

    def random_coords(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """A plausible random draw — gaussian, weighted toward the dominant
        axes, clamped to each axis' realizable range (port of randomVoice)."""
        rng = rng or np.random.default_rng()
        g = rng.standard_normal(self.k) * (0.5 + self.var_explained * 3.0)
        return np.clip(g, self.rng[:, 0], self.rng[:, 1]).astype(np.float32)

    # ───────────────────────── offsets ─────────────────────────
    def add_timbre(self, style: np.ndarray, amounts: dict[str, float]) -> np.ndarray:
        for e, a in amounts.items():
            r = self.emotion_full.get(e)
            if a and r is not None:
                style = style + a * r
        return style

    def add_mascfem(self, style: np.ndarray, alpha: float) -> np.ndarray:
        if alpha and self.mascfem is not None:
            style = style + alpha * self.mascfem
        return style

    # ───────────────────────── clone ─────────────────────────
    def style_from_ecapa(self, x: np.ndarray) -> np.ndarray:
        """1024-D ECAPA embedding → 256-D style: style = ym + (x − xm)·B."""
        if self.bridge is None:
            raise RuntimeError("voice_bridge.bin not loaded — clone unavailable")
        br = self.bridge
        return br["ym"] + (np.asarray(x, np.float32) - br["xm"]) @ br["B"]
