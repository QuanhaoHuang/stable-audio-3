"""RungDiT — medium DiT inference via a single multi-signature .tflite (dit_{prec}.tflite: static rung
subgraphs s<R> sharing weight buffers, built by build/build_dit.sh from stable_audio_3.models.dit).
Picks the SMALLEST rung R >= L and runs ONE full-length forward with the extra (R-L) KEYS masked out of
self-attention (attn_mask), so a render of real length L on rung R is EXACT (no SWA tiling — the DiT is a
single forward, unlike the decoder). On litert>=2.2.0 it loads ONE CompiledModel + XNNPACK weight cache so
the 10 rungs share one copy of the weights, and dispatches by rung -> peak RAM is the high-water mark of the
largest rung actually used, not the sum.

The 7 BAKED inputs match export_dit.DiTBaked.forward (args_0..6, in this order):
    x[1,256,R]  t[1]  t5_hidden[1,256,768]  t5_mask[1,256]  seconds[1]  local_add_cond[1,257,R]
    attn_mask[1,1,1,MEM+R]   (0 for the MEM+L valid keys, -1e9 for the R-L pad keys)
The conditioner (T5 padding + seconds) is baked in-graph. Batch is baked to 1, so CFG runs as a SEQUENTIAL
dual-pass (cond + uncond), exactly like the static-batch=1 TensorRT engine. Interface mirrors BakedDiT so
sa3_tflite.main() can swap it in: __call__(x, t, cross, gcond) -> v (or cfg-guided v)."""
from __future__ import annotations
import numpy as np

MEM = 64               # memory/register tokens prepended in-graph (must match export_dit.MEM)
COND_TOKENS = 256
COND_DIM = 768
LATENT_CH = 256


class RungDiT:
    """model_fn(x,t,cross,gcond)->v compatible with P.sample. cross/gcond are IGNORED (conditioning is
    baked in-graph, driven by the T5 outputs held here). cfg==1.0 -> one forward; cfg!=1.0 -> sequential
    cond+uncond dual-pass combined by cfg_fn (sa3_tflite._apply_cfg) in denoised space, with optional APG."""

    def __init__(self, path, L, t5_hidden, t5_mask, seconds, threads=8, cfg=1.0, apg=1.0,
                 null_hidden=None, null_mask=None, local_add_cond=None,
                 weight_cache_path="auto", cfg_fn=None, **_ignored):
        from ai_edge_litert.compiled_model import CompiledModel, Options
        from ai_edge_litert.cpu_options import CpuOptions
        path = str(path)
        wc = (path + ".xnnwc") if weight_cache_path == "auto" else weight_cache_path
        self.m = CompiledModel.from_file(path, options=Options(cpu_options=CpuOptions(
            num_threads=int(threads), xnnpack_weight_cache_path=str(wc) if wc else "")))
        # discover rung sizes from the file's s<N> signatures (any rung set just works)
        keys = {self.m.get_signature_by_index(i)["key"]: i for i in range(self.m.get_num_signatures())}
        self.sig = {int(k[1:]): v for k, v in keys.items() if k.startswith("s") and k[1:].isdigit()}
        if not self.sig:
            raise ValueError(f"{path}: no s<N> rung signatures found")
        self.sizes = sorted(self.sig)
        self.max_rung = self.sizes[-1]
        self._cfg_fn = cfg_fn
        self._slots = self._input_slots(path)          # semantic name -> buffer position (0..6)
        self._bufs = {}                                # si -> (in_bufs, out_bufs) (lazy per rung)
        self.set_conditioning(L, t5_hidden, t5_mask, seconds, cfg=cfg, apg=apg,
                              null_hidden=null_hidden, null_mask=null_mask, local_add_cond=local_add_cond)

    def _input_slots(self, path):
        """Map each of the 7 baked inputs to its CompiledModel buffer position. create_input_buffers()
        follows the SUBGRAPH inputs order == Interpreter.get_input_details() RAW order (NOT the signature
        alias order args_0..6) — validated max|Δ|=0 vs a name-fed run. Identify each by shape (5 unique);
        t vs seconds (both rank-1) split by tensor-name order (args_1=t < args_4=seconds). All rungs share
        one export graph so subgraph 0's order holds for every signature; _write re-checks via shape."""
        from ai_edge_litert.interpreter import Interpreter
        det = Interpreter(model_path=path).get_input_details()   # raw list order == buffer order
        slots, rank1 = {}, []
        for j, d in enumerate(det):
            shp = [int(s) for s in d["shape"]]
            if len(shp) == 4:
                slots["am"] = j
            elif len(shp) == 2:
                slots["t5m"] = j
            elif len(shp) == 3 and shp[1] == 257:
                slots["lac"] = j
            elif len(shp) == 3 and shp[2] == COND_DIM:
                slots["t5h"] = j
            elif len(shp) == 3 and shp[1] == LATENT_CH:
                slots["x"] = j
            elif len(shp) == 1:
                rank1.append((d["name"], j))
        rank1.sort()                                      # args_1=t before args_4=seconds
        if len(rank1) == 2:
            slots["t"], slots["sec"] = rank1[0][1], rank1[1][1]
        missing = {"x", "t", "t5h", "t5m", "sec", "lac", "am"} - set(slots)
        if missing:
            raise ValueError(f"{path}: could not map DiT inputs {missing} "
                             f"(shapes={[[int(s) for s in d['shape']] for d in det]})")
        return slots

    def _rung_ge(self, L):
        ge = [s for s in self.sizes if s >= L]
        if not ge:
            raise ValueError(f"requested DiT length {L} exceeds largest rung {self.max_rung}")
        return ge[0]

    def set_conditioning(self, L, t5_hidden, t5_mask, seconds, *, cfg=1.0, apg=1.0,
                         null_hidden=None, null_mask=None, local_add_cond=None, **_ignored):
        """(Re)bind per-generation conditioning. Picks the rung R>=L, builds the pad-key attn_mask, and
        writes the constant inputs (seconds, local_add_cond, attn_mask — and, for cfg==1, t5h/t5m) once
        into the rung's resident buffers; only x and t change per diffusion step."""
        self.L = int(L)
        self.R = self._rung_ge(self.L)
        self.si = self.sig[self.R]
        self.cfg = float(cfg); self.apg = float(apg)
        self.n_fwd = 0
        if self.si not in self._bufs:
            self._bufs[self.si] = (self.m.create_input_buffers(self.si), self.m.create_output_buffers(self.si))
        self.inb, self.outb = self._bufs[self.si]

        self.t5h = self._pad_t5(t5_hidden)
        self.t5m = t5_mask.astype(np.float32).reshape(1, COND_TOKENS)
        self.null_h = None if null_hidden is None else self._pad_t5(null_hidden)
        self.null_m = None if null_mask is None else null_mask.astype(np.float32).reshape(1, COND_TOKENS)
        # seconds + local_add_cond + attn_mask are constant across steps AND cfg branches -> resident.
        sec = np.array([np.float32(seconds)], np.float32)
        lac = (np.zeros((1, 257, self.R), np.float32) if local_add_cond is None
               else self._pad_len(local_add_cond.astype(np.float32), 257))
        am = np.zeros((1, 1, 1, MEM + self.R), np.float32)
        am[..., MEM + self.L:] = -1e9                     # mask the (R-L) pad KEYS -> exact at length L
        self._write("sec", sec); self._write("lac", lac); self._write("am", am)
        if self.cfg == 1.0:                               # single branch -> t5h/t5m resident too
            self._write("t5h", self.t5h); self._write("t5m", self.t5m)
        return self

    def _pad_t5(self, h):
        return h.astype(np.float32).reshape(1, COND_TOKENS, COND_DIM)

    def _pad_len(self, a, ch):
        """Pad a [1,ch,L] tensor up to [1,ch,R] (pad content is irrelevant: pad queries are trimmed and
        pad keys are masked out of attention — no cross-position leak except attention, which is masked)."""
        L = a.shape[2]
        if L == self.R:
            return np.ascontiguousarray(a)
        out = np.zeros((1, ch, self.R), np.float32)
        out[:, :, :L] = a[:, :, :self.R]
        return out

    def _write(self, name, arr):
        self.inb[self._slots[name]].write(np.ascontiguousarray(arr, np.float32))

    def _fwd(self, x, t, t5h=None, t5m=None):
        """One batch=1 forward on the active rung. x is [1,256,L] (padded to R); returns v [1,256,L]."""
        self._write("x", self._pad_len(x.astype(np.float32), LATENT_CH))
        self._write("t", np.array([np.float32(t)], np.float32))
        if t5h is not None:                               # CFG branches rebind t5h/t5m per pass
            self._write("t5h", t5h); self._write("t5m", t5m)
        self.m.run_by_index(self.si, self.inb, self.outb)
        self.n_fwd += 1
        v = np.asarray(self.outb[0].read(LATENT_CH * self.R, np.float32)).reshape(1, LATENT_CH, self.R)
        return v[:, :, :self.L].copy()

    def __call__(self, x, t, cross=None, gcond=None):
        if self.cfg == 1.0:
            return self._fwd(x, t)
        v_cond = self._fwd(x, t, self.t5h, self.t5m)
        v_uncond = self._fwd(x, t, self.null_h, self.null_m)
        if self._cfg_fn is None:
            raise RuntimeError("RungDiT needs cfg_fn (sa3_tflite._apply_cfg) for cfg != 1.0")
        return self._cfg_fn(x, t, v_cond, v_uncond, self.cfg, self.apg)
