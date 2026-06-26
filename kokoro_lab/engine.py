"""Engine — the staged Kokoro forward over the upstream `kokoro` KModel.

This reimplements KModel.forward_with_tokens step by step (kokoro/model.py)
so that, unlike the stock one-shot forward, it:

  * captures every intermediate as a named stage (the lab's "trace"), and
  * hands back a DecodeContext so the decoder back-half can be re-run from
    edited prosody (F0 / energy / duration) WITHOUT re-running the front-end —
    exactly brosoundml's prepare_decode_context + decode_from.

The Kokoro pipeline (matching the stage names in kokoro-lab/lib/state.js):

    phonemes → bert_dur → d_en → (predictor) → pred_dur → F0_pred / N_pred
             → t_en → asr → (decoder) → audio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class Trace:
    """Host copies of the pipeline intermediates, for visualization. Each grid
    is row-major (row = channel/feature). Mirrors KokoroTrace in kokoro.h."""
    phoneme_ids: list[int] = field(default_factory=list)
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    audio: Optional[np.ndarray] = None
    sample_rate: int = 24000

    def add(self, name: str, t: torch.Tensor) -> None:
        a = t.detach().squeeze().float().cpu().numpy()
        self.stages[name] = a


@dataclass
class DecodeContext:
    """Everything needed to re-decode the back half from edited prosody. The
    front-end (bert/predictor-encoder/text-encoder) is already spent; only the
    alignment, F0Ntrain and decoder are re-run on a retime."""
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    text_mask: torch.Tensor
    d: torch.Tensor          # predictor-encoder output (for re-timing → en)
    t_en: torch.Tensor       # text-encoder content (for re-timing → asr)
    s: torch.Tensor          # ref_s[:, 128:] — predictor style
    ref_dec: torch.Tensor    # ref_s[:, :128] — decoder style
    pred_dur: torch.Tensor   # (L,) predicted frames per phoneme
    asr: torch.Tensor        # t_en @ aln (current timing)
    f0: torch.Tensor         # current F0_pred
    n: torch.Tensor          # current N_pred


class KokoroLab:
    def __init__(self, model, device: Optional[str] = None):
        """`model` is a kokoro.KModel (already on its device)."""
        self.model = model
        self.device = device or str(next(model.parameters()).device)

    # ───────────────────────── construction ─────────────────────────
    @classmethod
    def load(cls, device: Optional[str] = None, config: Optional[str] = None,
             weights: Optional[str] = None):
        """Load the stock Kokoro-82M (downloads from HF on first run)."""
        # Import KModel directly from the submodule, not the package root —
        # kokoro/__init__.py pulls in KPipeline → misaki → spacy, which is only
        # needed for g2p. Loading/running the model itself needs none of it.
        from kokoro.model import KModel
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        km = KModel(config=config, model=weights).to(device).eval()
        return cls(km, device)

    # ───────────────────────── phonemize ─────────────────────────
    def make_g2p(self, lang_code: str = "a"):
        """A 'quiet' KPipeline (no model) used only for grapheme→phoneme."""
        from kokoro import KPipeline
        return KPipeline(lang_code=lang_code, model=False)

    @staticmethod
    def phonemize(g2p, text: str) -> str:
        """Return the phoneme string for `text` (first chunk concatenated)."""
        out = []
        for _, ps, _ in g2p(text):
            out.append(ps)
        return "".join(out)

    def ids_for(self, phonemes: str) -> list[int]:
        vocab = self.model.vocab
        return [vocab[p] for p in phonemes if p in vocab and vocab[p] is not None]

    # ───────────────────────── staged forward ─────────────────────────
    @torch.no_grad()
    def synthesize(self, phonemes: str, ref_s: np.ndarray, speed: float = 1.0,
                   trace: bool = True):
        """Full forward. Returns (audio: np.float32, Trace, DecodeContext).

        `ref_s` is the 256-D designed style vector (see VoiceSpace)."""
        m = self.model
        dev = self.device
        ids = self.ids_for(phonemes)
        if not ids:
            raise ValueError("no phonemes for that text")

        input_ids = torch.LongTensor([[0, *ids, 0]]).to(dev)
        input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(dev)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0)
        text_mask = text_mask.expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(dev)

        ref = torch.as_tensor(np.asarray(ref_s, np.float32)).reshape(1, -1).to(dev)
        s = ref[:, 128:]
        ref_dec = ref[:, :128]

        # ── front-end: plBERT + predictor encoder ──
        bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        # ── duration ──
        x, _ = m.predictor.lstm(d)
        duration = m.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        # ── alignment, prosody, content ──
        aln = self._alignment(input_ids.shape[1], pred_dur, dev)
        en = d.transpose(-1, -2) @ aln
        f0, n = m.predictor.F0Ntrain(en, s)
        t_en = m.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ aln

        # ── decoder ──
        audio = m.decoder(asr, f0, n, ref_dec).squeeze()
        audio_np = audio.float().cpu().numpy()

        tr = Trace(phoneme_ids=ids, audio=audio_np, sample_rate=24000)
        if trace:
            tr.add("bert_dur", bert_dur)                       # L × 768
            tr.add("d_en", d_en)                               # 512 × L (prosody)
            tr.add("t_en", t_en)                               # 512 × L (content)
            tr.stages["pred_dur"] = pred_dur.detach().cpu().numpy().astype(np.int32)
            tr.add("F0_pred", f0)                              # frame-rate pitch
            tr.add("N_pred", n)                                # frame-rate energy
            tr.add("asr", asr)                                 # 512 × T

        ctx = DecodeContext(
            input_ids=input_ids, input_lengths=input_lengths, text_mask=text_mask,
            d=d, t_en=t_en, s=s, ref_dec=ref_dec, pred_dur=pred_dur,
            asr=asr, f0=f0, n=n,
        )
        return audio_np, tr, ctx

    # ───────────────────────── general re-decode (any timing + contours) ─────────────────────────
    @torch.no_grad()
    def decode_full(self, ctx: DecodeContext, dur, f0, n) -> np.ndarray:
        """Re-decode from an arbitrary per-phoneme duration set and F0/N contours.
        Length-regulates t_en onto `dur` (→ asr), then runs the decoder. `f0`/`n`
        must be length 2·sum(dur). This is the one path used for every edit —
        curve draws (dur unchanged) and re-timing alike."""
        dev = self.device
        dur_t = torch.as_tensor(np.asarray(dur).ravel().astype("int64"), device=dev).clamp(min=1)
        aln = self._alignment(ctx.input_ids.shape[1], dur_t, dev)
        asr = ctx.t_en @ aln
        f0t = torch.as_tensor(np.asarray(f0, np.float32).ravel(), device=dev).reshape(1, -1)
        nt = torch.as_tensor(np.asarray(n, np.float32).ravel(), device=dev).reshape(1, -1)
        audio = self.model.decoder(asr, f0t, nt, ctx.ref_dec).squeeze()
        return audio.float().cpu().numpy()

    @staticmethod
    def resample_by_dur(src, src_dur, dst_dur) -> np.ndarray:
        """Resample a frame-rate contour from one per-phoneme duration set to
        another, preserving each phoneme's contour SHAPE while restretching its
        time span. `src` is at 2× frame rate (len 2·sum(src_dur)); returns length
        2·sum(dst_dur). Port of kokoro-lab/lib/edit.js resampleByDur."""
        src = np.asarray(src, np.float32).ravel()
        src_dur = np.asarray(src_dur).astype(int).ravel()
        dst_dur = np.asarray(dst_dur).astype(int).ravel()
        dst = np.zeros(2 * int(dst_dur.sum()), np.float32)
        s_off = d_off = 0
        for l in range(len(src_dur)):
            s_len, d_len = 2 * int(src_dur[l]), 2 * int(dst_dur[l])
            s0, d0 = 2 * s_off, 2 * d_off
            if s_len > 0 and d_len > 0:
                if d_len == 1:
                    dst[d0] = src[s0]
                else:
                    sp = (np.arange(d_len) / (d_len - 1)) * (s_len - 1)
                    i0 = np.floor(sp).astype(int)
                    i1 = np.minimum(s_len - 1, i0 + 1)
                    fr = sp - i0
                    dst[d0:d0 + d_len] = src[s0 + i0] * (1 - fr) + src[s0 + i1] * fr
            s_off += int(src_dur[l]); d_off += int(dst_dur[l])
        return dst

    # ───────────────────────── decode-from-stage ─────────────────────────
    @torch.no_grad()
    def decode(self, ctx: DecodeContext, f0: Optional[torch.Tensor] = None,
               n: Optional[torch.Tensor] = None) -> np.ndarray:
        """Re-run only the decoder back-half with (optionally) edited F0/energy
        at the current timing — brosoundml's decode_from."""
        f0 = ctx.f0 if f0 is None else self._as_tensor(f0, ctx.f0)
        n = ctx.n if n is None else self._as_tensor(n, ctx.n)
        audio = self.model.decoder(ctx.asr, f0, n, ctx.ref_dec).squeeze()
        return audio.float().cpu().numpy()

    @torch.no_grad()
    def retime(self, ctx: DecodeContext, pred_dur: torch.Tensor):
        """Rebuild the alignment from new per-phoneme durations and recompute
        the timing-dependent stages (en→F0Ntrain, asr). Mutates and returns
        ctx so a subsequent decode() uses the new timing."""
        dev = self.device
        pred_dur = torch.as_tensor(pred_dur, device=dev).long().clamp(min=1)
        aln = self._alignment(ctx.input_ids.shape[1], pred_dur, dev)
        en = ctx.d.transpose(-1, -2) @ aln
        f0, n = self.model.predictor.F0Ntrain(en, ctx.s)
        ctx.pred_dur = pred_dur
        ctx.asr = ctx.t_en @ aln
        ctx.f0, ctx.n = f0, n
        return ctx

    # ───────────────────────── emotion (parametric prosody) ─────────────────────────
    @torch.no_grad()
    def apply_emotion(self, ctx: DecodeContext, emo) -> tuple[np.ndarray, Trace]:
        """Apply a Tier-0 VAD emotion: rate retimes the alignment, then pitch
        register/range and energy reshape the freshly-predicted contours, and
        the back-half is re-decoded. Returns (audio, updated-prosody Trace)."""
        rate = emo.rate_scale()
        if abs(rate - 1.0) > 1e-3:
            new_dur = torch.round(ctx.pred_dur.float() / rate).clamp(min=1).long()
            self.retime(ctx, new_dur)
        f0_np, n_np = emo.transform_contours(
            ctx.f0.detach().cpu().numpy().squeeze(),
            ctx.n.detach().cpu().numpy().squeeze(),
        )
        audio = self.decode(ctx, f0=f0_np, n=n_np)
        tr = Trace()
        tr.stages["F0_pred"] = f0_np
        tr.stages["N_pred"] = n_np
        tr.stages["pred_dur"] = ctx.pred_dur.detach().cpu().numpy().astype(np.int32)
        tr.audio = audio
        return audio, tr

    # ───────────────────────── helpers ─────────────────────────
    @staticmethod
    def _alignment(n_tokens: int, pred_dur: torch.Tensor, dev) -> torch.Tensor:
        """Build the (1, L, T) hard monotonic alignment from per-token frame
        counts — verbatim from kokoro/model.py."""
        pred_dur = pred_dur.reshape(-1)
        indices = torch.repeat_interleave(
            torch.arange(n_tokens, device=dev), pred_dur)
        aln = torch.zeros((n_tokens, indices.shape[0]), device=dev)
        aln[indices, torch.arange(indices.shape[0], device=dev)] = 1
        return aln.unsqueeze(0)

    def _as_tensor(self, arr, like: torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            t = arr.to(self.device, torch.float32)
        else:
            t = torch.as_tensor(np.asarray(arr, np.float32), device=self.device)
        return t.reshape(like.shape)
