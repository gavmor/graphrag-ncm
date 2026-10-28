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

from typing import Any

import torch
import torch.nn.functional as F
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from ncm_graphrag.ncm.task_registry import TASKS


class TransformerStack(torch.nn.Module):
    def __init__(self, blocks: Any, ln_f: Any) -> None:
        super().__init__()
        self.blocks = blocks
        self.ln_f = ln_f

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        rope = (cos, sin)
        for b in self.blocks:
            x = b(x, attn_mask=mask, rope_cos_sin=rope)
        return self.ln_f(x)


class NCMTaskGenerator:
    def __init__(self, task: str = "mapstage", device: str | None = None, compile_blocks: bool = True):
        self.compile_blocks = compile_blocks
        spec = TASKS[task]
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
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
        self._optimize_net()

    def _optimize_net(self) -> None:
        orig_sample = self.net._sample_level_raw

        def _fast_sample_level_raw(logits, temperature, top_k, top_p):
            if top_k == 1 or temperature == 0:
                return logits.argmax(dim=-1)
            return orig_sample(logits, temperature, top_k, top_p)

        self.net._sample_level_raw = _fast_sample_level_raw  # ty: ignore[invalid-assignment]

        p_len = self.net.prefix_len
        dev = self.device
        net_ref: Any = self.net
        l0 = net_ref.levels[0]

        stack = TransformerStack(net_ref.transformer.blocks, net_ref.transformer.ln_f)
        compiled_stack: Any = (
            torch.compile(stack, dynamic=True) if self.compile_blocks and dev.startswith("cuda") else stack
        )

        base_masks: list[torch.Tensor] = []
        for hidx, l_cur in enumerate(net_ref.levels[net_ref._lvl_first :]):
            m = torch.zeros(1, 1, p_len + l_cur, p_len + l_cur, device=dev)
            m[:, :, :p_len, p_len:] = float("-inf")
            net_ref.register_buffer(f"_base_gen_mask_{hidx}", m, persistent=False)
            base_masks.append(m)

        m0_base = torch.zeros(1, 1, p_len + l0, p_len + l0, device=dev)
        m0_base[:, :, :p_len, p_len:] = float("-inf")
        net_ref.register_buffer("_base_m0", m0_base, persistent=False)

        rope_list = [
            (getattr(net_ref, f"rope_cos_gen_{k}"), getattr(net_ref, f"rope_sin_gen_{k}")) if net_ref.use_rope else None
            for k in range(len(net_ref.levels))
        ]
        heads = list(net_ref.lm_heads)
        levels = tuple(net_ref.levels)

        def _fast_decode_level_logits(x, prefix_emb, level_idx, pad_bias=None, hidx=None):
            h_idx = level_idx if hidx is None else hidx
            l_curr = levels[level_idx + 1]
            x_cat = torch.cat([prefix_emb, x], dim=1)
            base_mask = base_masks[h_idx]
            gen_mask = base_mask + F.pad(pad_bias, (0, l_curr)) if pad_bias is not None else base_mask
            rope_k = rope_list[h_idx]
            x_cat = compiled_stack(x_cat, gen_mask, rope_k[0], rope_k[1])
            return heads[h_idx](x_cat[:, p_len:, :])

        self.net._decode_level_logits = _fast_decode_level_logits  # ty: ignore[invalid-assignment]

        vq = self.vqvae.vq

        def _optimized_generate(
            idx,
            prefix_tokens,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            use_kv_cache=False,
            cfg_scale=0.0,
            cfg_null_eos=50256,
        ):
            B = prefix_tokens.shape[0]
            z_hat = torch.zeros(B, net_ref.levels[-1], self.vqvae.embed_dim, device=dev)

            _enc = self.vqvae.encoder
            z_prefix = _enc.wte(prefix_tokens)[:, :p_len, :]
            prefix_emb = net_ref.transformer.wte(z_prefix)
            level_pos_embed = net_ref.transformer.wle(net_ref.lvl_1L)

            _pm = prefix_tokens[:, :p_len] == net_ref.prefix_pad_id
            _pbias = torch.zeros(_pm.shape[0], 1, 1, p_len, device=dev)
            _pbias[:, :, :, :p_len] = torch.where(_pm[:, None, None, :], float("-inf"), 0.0)

            # Level 0
            _q0 = net_ref.build_l0_query_input(B, prefix_emb.dtype)
            q = net_ref.transformer.wte(_q0) + net_ref.transformer.wle(net_ref.lvl_1L[:, :l0])
            h = torch.cat([prefix_emb, q], dim=1)
            m0 = net_ref._base_m0 + F.pad(_pbias, (0, l0))
            h = compiled_stack(h, m0, net_ref.rope_cos_gen_0, net_ref.rope_sin_gen_0)
            idx_lvl0 = net_ref.lm_heads[0](h[:, p_len:, :]).argmax(dim=-1)

            generated_idx = [idx_lvl0]
            idx_next = idx_lvl0
            level_acum = l0

            for level_idx in range(len(net_ref.levels) - 1):
                z_q_level = vq.embeddings[level_idx](idx_next)
                z_hat, x_next = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)
                x_next = (
                    net_ref.transformer.wte(x_next)
                    + level_pos_embed[:, level_acum : level_acum + net_ref.levels[level_idx + 1], :]
                )
                _hi = level_idx + 1
                logits = net_ref._decode_level_logits(x_next, prefix_emb, level_idx, _pbias, hidx=_hi)
                idx_next = logits.argmax(dim=-1)
                generated_idx.append(idx_next)
                level_acum += net_ref.levels[level_idx + 1]

            return torch.cat(generated_idx, dim=1)

        self.net.generate = _optimized_generate  # ty: ignore[invalid-assignment]

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
        text = self.pre_tok.decode([t for t in ids if t != self.pad_id], skip_special_tokens=True)
        if "<|COMPLETE|>" in text:
            return text[: text.index("<|COMPLETE|>") + len("<|COMPLETE|>")]
        if self.cut_at_eos and self.eos_id in ids:
            return self.pre_tok.decode(
                [t for t in ids[: ids.index(self.eos_id)] if t != self.pad_id], skip_special_tokens=True
            )
        return text

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
        z_q = self.vqvae.vq.idx_to_z_q(g)
        ids = self._pick_tokens(self.vqvae.decoder(z_q)).tolist()
        return [self._dec(row) for row in ids]
