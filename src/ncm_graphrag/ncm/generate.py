"""Loads a trained tokenizer+generator pair for a registered task and exposes
`generate(text)` / `generate_best_of_n(text)`.

Decode logic (`_pick_tokens`/`_dec`) is copied from `scripts/eval_shard.py` to match
the released checkpoints' decoding quality: greedy argmax is degenerate/repetitive at
this param count (see docs/constraints.md), so the frequency-penalty + adjacency-
penalty greedy loop is required for non-garbage output.

The LM1B-style EOS-off-by-one bug noted in the task brief (generation seeds the
sequence with EOS as a position-0 BOS, so `dec()`'s cut-at-first-EOS returns empty
strings unless position 0 is skipped) is specific to `predict_level0` unconditional
sampling (`NextConceptTransformer.generate(idx=None, ...)` with `query_level0=False`).
Both registered tasks use `query_level0=True` (conditional generation via a learned
query slot, like the released xsum checkpoint) -- that code path is never hit here,
verified by reading `NextConceptTransformer.generate()`'s branching rather than
assumed. `_dec()` below is otherwise the unmodified upstream logic.
"""

import torch
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from ncm_graphrag.ncm.task_registry import TASKS


class NCMTaskGenerator:
    def __init__(self, task: str = "mapstage", device: str | None = None):
        spec = TASKS[task]
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok_cfg = spec.build_tokenizer_cfg()
        model_cfg = spec.build_model_cfg()
        gen_cfg = spec.build_generation_cfg()
        self.is_well_formed = spec.is_well_formed

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

    def _prefix_tensor(self, text: str, n: int = 1) -> torch.Tensor:
        ids = self.pre_tok.encode(text)
        length = min(len(ids), self.prefix_len)
        x = torch.full((n, self.prefix_len), self.pad_id, dtype=torch.long, device=self.device)
        if length > 0:
            x[:, :length] = torch.tensor(ids[:length], dtype=torch.long, device=self.device)
        return x

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

    @torch.inference_mode()
    def generate(self, prefix_text: str) -> str:
        """One conditional generation for this task: bounded prefix text in, bounded
        task-shaped text out (JSON for mapstage, graphrag's delimiter-tuple grammar
        for extraction).

        Deterministic (temperature=1.0, top_k=1 at the level-code sampling stage in
        `net.generate()`, plus the greedy `_pick_tokens` penalty decode at the final
        token stage) -- repeated calls on the same input return the same output. See
        `generate_best_of_n` for the non-degenerate multi-sample path.
        """
        return self._decode_batch(prefix_text, n=1, temp=self.temp, top_k=self.top_k, top_p=self.top_p)[0]

    @torch.inference_mode()
    def generate_best_of_n(self, prefix_text: str, n: int = 8, sample_temp: float = 0.9, sample_top_k: int = 16) -> str:
        """Sample `n` candidates in one batched forward pass; among those passing this
        task's well-formedness check (`task_registry.TaskSpec.is_well_formed`), return
        the one with the highest total word-overlap similarity to the others (Minimum
        Bayes Risk selection -- the candidate "closest to consensus" is a cheap proxy
        for "most likely correct" when no reference is available). Falls back to the
        plain deterministic `generate()` output if none pass well-formedness.

        Both the sampling budget and the MBR step address a real, measured gap: on the
        extraction task, well-formedness alone doesn't guarantee correct *content* --
        entity names and descriptions can still be garbled in an otherwise
        structurally-valid candidate (see docs/architecture.md). Picking the
        highest-consensus candidate measurably improves on "first valid" without any
        extra training. Using the model's own cheap parallel forward pass as the
        reranking budget matches the task brief's observation that both tasks already
        generate many short outputs per query/document.
        """
        candidates = self._decode_batch(prefix_text, n=n, temp=sample_temp, top_k=sample_top_k, top_p=1.0)
        valid = [c for c in candidates if self.is_well_formed(c)]
        if not valid:
            return self.generate(prefix_text)
        if len(valid) == 1:
            return valid[0]

        def word_set(text: str) -> set[str]:
            return set(text.lower().split())

        def jaccard(a: set[str], b: set[str]) -> float:
            return len(a & b) / max(len(a | b), 1)

        word_sets = [word_set(c) for c in valid]
        scores = [
            sum(jaccard(ws, other) for j, other in enumerate(word_sets) if j != i) for i, ws in enumerate(word_sets)
        ]
        return valid[max(range(len(valid)), key=lambda i: scores[i])]

    @torch.inference_mode()
    def _decode_batch(self, prefix_text: str, n: int, temp: float, top_k: int, top_p: float) -> list[str]:
        x = self._prefix_tensor(prefix_text, n)
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
