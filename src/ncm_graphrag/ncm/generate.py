"""Loads the trained map-stage tokenizer+generator pair and exposes `generate(text)`.

Decode logic (`_pick_tokens`/`_dec`) is copied from `scripts/eval_shard.py` to match
the released checkpoints' decoding quality: greedy argmax is degenerate/repetitive at
this param count (see docs/constraints.md), so the frequency-penalty + adjacency-
penalty greedy loop is required for non-garbage output.

The LM1B-style EOS-off-by-one bug noted in the task brief (generation seeds the
sequence with EOS as a position-0 BOS, so `dec()`'s cut-at-first-EOS returns empty
strings unless position 0 is skipped) is specific to `predict_level0` unconditional
sampling (`NextConceptTransformer.generate(idx=None, ...)` with `query_level0=False`).
This model uses `query_level0=True` (conditional generation via a learned query slot,
like the released xsum checkpoint) -- that code path is never hit here, verified by
reading `NextConceptTransformer.generate()`'s branching rather than assumed. `_dec()`
below is otherwise the unmodified upstream logic.
"""

import torch
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from ncm_graphrag.ncm.config import build_generation_cfg, build_model_cfg, build_tokenizer_cfg


class MapStageGenerator:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok_cfg = build_tokenizer_cfg()
        model_cfg = build_model_cfg()
        gen_cfg = build_generation_cfg()

        self.pre_tok = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
        self.prefix_len = int(model_cfg.prefix_len)
        self.pad_id = self.pre_tok.pad_token_id
        self.eos_id = self.pre_tok.eos_token_id

        self.vqvae = VQVAE(**tok_cfg.vqvae).to(self.device)
        tok_state = torch.load(tok_cfg.save_path, map_location="cpu", weights_only=False)
        self.vqvae.load_state_dict(tok_state["model"], strict=True)
        self.vqvae.eval()

        self.net = NextConceptTransformer(vqvae_local=self.vqvae, config=model_cfg).to(self.device)
        gen_state = torch.load(model_cfg.save_path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(gen_state["model"], strict=True)
        self.net.eval()

        self.temp = float(gen_cfg.temperature)
        self.top_k = int(gen_cfg.top_k)
        self.top_p = float(gen_cfg.top_p)
        self.cfg_scale = float(gen_cfg.cfg_scale)
        self.freq = float(gen_cfg.frequency_penalty)
        self.mu = float(gen_cfg.frequency_deficit)
        adj = str(gen_cfg.adjacency_penalty)
        self.adj = [float(v) for v in adj.split(",")] if adj else None
        self.cut_at_eos = bool(gen_cfg.cut_at_eos)

    def _prefix_ids(self, text: str) -> list[int]:
        ids = self.pre_tok.encode(text)
        return (ids[: self.prefix_len] + [self.pad_id] * self.prefix_len)[: self.prefix_len]

    def _pick_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy decode with frequency + adjacency penalties (copied from eval_shard.py's
        pick_tokens; required for non-garbage output -- see module docstring)."""
        if self.freq <= 0 and not self.adj:
            return logits.argmax(-1)
        bsz, steps, vocab = logits.shape
        lg = logits.double()
        al = self.adj[0] if self.adj else 0.0
        aw = int(self.adj[1]) if self.adj and len(self.adj) > 1 else 1
        ad = self.adj[2] if self.adj and len(self.adj) > 2 else 1.0
        cnt = torch.zeros(bsz, vocab, dtype=torch.float64, device=lg.device)
        out = torch.empty(bsz, steps, dtype=torch.long, device=lg.device)
        hist: list[torch.Tensor] = []
        for t in range(steps):
            step_logits = lg[:, t, :].clone()
            if self.freq > 0:
                step_logits -= self.freq * cnt
            for j, h in enumerate(reversed(hist[-aw:])):
                step_logits.scatter_add_(
                    1, h[:, None], torch.full((bsz, 1), -al * ad**j, dtype=torch.float64, device=lg.device)
                )
            picked = step_logits.argmax(-1)
            out[:, t] = picked
            hist.append(picked)
            cnt.scatter_add_(1, picked[:, None], torch.ones(bsz, 1, dtype=torch.float64, device=lg.device))
        return out

    def _dec(self, ids: list[int]) -> str:
        ids = list(ids)
        if self.cut_at_eos and self.eos_id in ids:
            ids = ids[: ids.index(self.eos_id)]
        return self.pre_tok.decode([t for t in ids if t not in (self.eos_id, self.pad_id)], skip_special_tokens=True)

    @torch.no_grad()
    def generate(self, prefix_text: str) -> str:
        """One conditional map-stage generation: "report + query" text in, JSON text out.

        Deterministic (temperature=1.0, top_k=1 at the level-code sampling stage in
        `net.generate()`, plus the greedy `_pick_tokens` penalty decode at the final
        token stage) -- repeated calls on the same input return the same output. See
        `generate_best_of_n` for the non-degenerate multi-sample path.
        """
        return self._decode_batch(prefix_text, n=1, temp=self.temp, top_k=self.top_k, top_p=self.top_p)[0]

    @torch.no_grad()
    def generate_best_of_n(self, prefix_text: str, n: int = 8, sample_temp: float = 0.9, sample_top_k: int = 16) -> str:
        """Sample `n` candidates in one batched forward pass and return the first that
        parses as JSON matching graphrag's expected `{"points":[{"description","score"}]}`
        shape, falling back to the plain deterministic `generate()` output if none do.

        Addresses the known single-sample quality gap noted in docs/constraints.md
        (greedy decode at this param count is rough -- ~57% of single-shot outputs are
        valid JSON in held-out smoke testing) by using the model's own cheap parallel
        forward pass as the reranking budget, per the task brief's observation that a
        map-stage caller already pays for many short generations per query.
        """
        import json as _json

        candidates = self._decode_batch(prefix_text, n=n, temp=sample_temp, top_k=sample_top_k, top_p=1.0)
        for cand in candidates:
            try:
                parsed = _json.loads(cand)
                points = parsed.get("points")
                if isinstance(points, list) and all("description" in p and "score" in p for p in points):
                    return cand
            except (ValueError, AttributeError):
                continue
        return self.generate(prefix_text)

    @torch.no_grad()
    def _decode_batch(self, prefix_text: str, n: int, temp: float, top_k: int, top_p: float) -> list[str]:
        prefix_ids = self._prefix_ids(prefix_text)
        x = torch.tensor([prefix_ids] * n, device=self.device)
        g = self.net.generate(
            None,
            x,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=self.cfg_scale,
            cfg_null_eos=self.eos_id,
        )
        _, zl = self.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        ids = self._pick_tokens(self.vqvae.decoder(zl[-1])).tolist()
        return [self._dec(row) for row in ids]
