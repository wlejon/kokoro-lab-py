"""Tier-0 emotion — a NumPy port of kokoro-lab's emotion.js.

No model and no training: emotion here is the *prosody* slice only — pitch
register/range, energy, and speaking rate, as closed-form transforms of the
predictor's own F0/N/duration output. The edited contours are re-decoded
through the same decoder back-half the manual editor uses (see engine.decode).

Arousal is the dominant, reliable prosodic axis; valence and dominance leave
only a weak prosodic trace (their main signature is timbre), so their gains are
deliberately small — this is the prosodic component of affect, not all of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class Emotion:
    v: float = 0.0  # valence   [-1, 1]
    a: float = 0.0  # arousal   [-1, 1]
    d: float = 0.0  # dominance [-1, 1]

    def active(self) -> bool:
        return self.v != 0.0 or self.a != 0.0 or self.d != 0.0

    # gain laws, verbatim from emotion.js
    def pitch_semis(self) -> float:
        return 2.5 * self.a + 1.0 * self.v - 1.5 * self.d

    def range_scale(self) -> float:
        return _clamp(1 + 0.45 * self.a + 0.20 * self.v - 0.15 * self.d, 0.5, 1.8)

    def energy_scale(self) -> float:
        return _clamp(1 + 0.35 * self.a + 0.20 * self.d, 0.5, 1.7)

    def rate_scale(self) -> float:
        return _clamp(1 + 0.30 * self.a - 0.12 * self.d, 0.6, 1.7)  # >1 = faster

    def transform_contours(self, f0: np.ndarray, n: np.ndarray):
        """Shift register + expand/contract pitch range in the log (musical)
        domain, anchored on the contour's own voiced geometric mean; unvoiced
        frames (F0≈0) stay unvoiced. Energy scales multiplicatively. Returns
        fresh arrays the same length/timing as the input."""
        f0 = np.asarray(f0, np.float32)
        n = np.asarray(n, np.float32)
        shift = 2.0 ** (self.pitch_semis() / 12.0)
        rng = self.range_scale()
        e_scale = self.energy_scale()

        voiced = f0 > 1e-3
        out_f0 = f0.copy()
        if voiced.any():
            mean_log = float(np.log(f0[voiced]).mean())
            out_f0[voiced] = np.clip(
                np.exp(mean_log + (np.log(f0[voiced]) - mean_log) * rng) * shift,
                0.0, 1000.0,
            )
        out_n = np.maximum(0.0, n * e_scale)
        return out_f0.astype(np.float32), out_n.astype(np.float32)
