"""kokoro-lab-py — a small, faithful Python reproduction of broworkshop's
kokoro-lab, built on the upstream `kokoro` pip package.

The lab splits into two layers:

  * voicespace / emotion — pure NumPy math over four small artifacts
    (voice_basis.json, emotion_basis.json, masc_fem_basis.json,
    voice_bridge.bin). No model. This is the bulk of the "controls".
  * engine — a staged Kokoro forward (KModel) that exposes every
    intermediate (the trace) and can re-decode just the back half from
    edited prosody. This is the only part that needs the model.

See README.md for the mapping back to the JS lab.
"""

from .voicespace import VoiceSpace
from .engine import KokoroLab, Trace, DecodeContext

__all__ = ["VoiceSpace", "KokoroLab", "Trace", "DecodeContext"]
