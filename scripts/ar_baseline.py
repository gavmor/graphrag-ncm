"""Autoregressive control for the Multi-News drafter: same data, same sample budget, matched size.

    NCM_CORPORA_ROOT=... uv run python scripts/ar_baseline.py train   # ~12000 steps x batch 128
    NCM_CORPORA_ROOT=... uv run python scripts/ar_baseline.py eval 60 /tmp/ar   # metrics + judge input

A decoder-only transformer, 6 layers x 384 wide x 6 heads (about 10.6M non-embedding parameters vs the
NCM generator's 10.9M), GPT-2 vocabulary with tied embeddings, trained on  <source> <EOS> <target> <EOS>
with loss on the target only. Greedy decoding, no repetition penalty (the fair comparison to the NCM's
own greedy decode). Purpose: separate "the NCM is worse" from "this scale and data is too little".
"""

import importlib
import json
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

importlib.import_module("ncm_graphrag.ncm._ncm_repo")  # puts the vendored ncm repo on sys.path

from tokenization.hf_tokenizer import HFTokenizer  # noqa: E402

from ncm_graphrag.data import report_multinews as data  # noqa: E402

CKPT = "/home/user/Desktop/gavmor/ncm-graphrag/artifacts/ar_baseline.pt"
D, LAYERS, HEADS, VOCAB, MAXLEN = 384, 6, 6, 50304, 288 + 1 + 48 + 1
BATCH, STEPS, LR = 128, 12000, 6e-4
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
EOS, PAD = tok.eos_token_id, tok.pad_token_id


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc, self.out = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (z.view(b, t, HEADS, D // HEADS).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.1 if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, D)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wte, self.wpe = nn.Embedding(VOCAB, D), nn.Embedding(MAXLEN, D)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln = nn.LayerNorm(D)

    def forward(  # ty: ignore[invalid-mutable-override]
        self, ids: torch.Tensor, select: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Logits for every position, or only where `select` is True (flattened): the full
        (batch x length x 50k) logit tensor is ~9 GB at batch 128, and the loss needs the target tokens only."""
        x = self.wte(ids) + self.wpe(torch.arange(ids.shape[1], device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        if select is not None:
            x = x[select]
        return x @ self.wte.weight.T  # tied output head


def encode(ex: "data.DraftExample") -> tuple[list[int], int]:
    src = tok.encode(ex.prefix_text)[:288] + [EOS]
    return src + tok.encode(ex.target_text)[:48] + [EOS], len(src)


def batch(rows: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), width), PAD, dtype=torch.long)
    lab = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (seq, n_src) in enumerate(rows):
        ids[i, : len(seq)] = torch.tensor(seq)
        lab[i, n_src : len(seq)] = torch.tensor(seq[n_src:])
    return ids.to(dev), lab.to(dev)


def train() -> None:
    torch.manual_seed(1)
    rows = [encode(e) for e in data.generate_examples(40000, 1)]
    held = [encode(e) for e in data.benchmark_examples()[:256]]
    model = GPT().to(dev)
    n_total = sum(p.numel() for p in model.parameters())
    non_embed = n_total - VOCAB * D - MAXLEN * D
    print(f"params: {n_total / 1e6:.2f}M total, {non_embed / 1e6:.2f}M non-embedding", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(0)
    for step in range(STEPS):
        lr = LR * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        ids, lab = batch([rows[rng.randrange(len(rows))] for _ in range(BATCH)])
        target = lab[:, 1:]
        loss = F.cross_entropy(model(ids[:, :-1], target != -100), target[target != -100])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == STEPS - 1:
            model.eval()
            with torch.no_grad():
                h_ids, h_lab = batch(held[:128])
                h_target = h_lab[:, 1:]
                h_loss = F.cross_entropy(
                    model(h_ids[:, :-1], h_target != -100), h_target[h_target != -100], ignore_index=-100
                )
            model.train()
            print(f"step {step:5d}  loss {loss.item():.4f}  HELDOUT loss {h_loss.item():.4f}", flush=True)
    torch.save(model.state_dict(), CKPT)
    print("saved", CKPT)


@torch.no_grad()
def greedy(model: GPT, ex: "data.DraftExample") -> str:
    src = tok.encode(ex.prefix_text)[:288] + [EOS]
    ids = torch.tensor([src], device=dev)
    for _ in range(48):
        nxt = model(ids)[0, -1].argmax().item()
        if nxt == EOS:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    return tok.decode(ids[0, len(src) :].tolist(), skip_special_tokens=True).strip()


def evaluate(n_judge: int, prefix: str) -> None:
    model = GPT().to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    model.eval()
    bench = data.benchmark_examples()
    outs = [greedy(model, e) for e in bench]
    n = len(bench)
    r1 = sum(data.rouge1_f(o, e.target_text) for o, e in zip(outs, bench, strict=False)) / n
    gr = sum(data.grounding(o, e.prefix_text) for o, e in zip(outs, bench, strict=False)) / n
    en = [data.entnum_grounding(o, e.prefix_text) for o, e in zip(outs, bench, strict=False)]
    print(
        f"AR draft_rouge1={100 * r1:.2f} grounding={100 * gr:.2f} "
        f"entnum_grounding={100 * sum(a for a, _ in en) / max(sum(b for _, b in en), 1):.2f}"
    )
    idx = list(range(n))
    random.Random(2).shuffle(idx)  # same 60 sources as scripts/judge_drafter.py
    idx = idx[:n_judge]
    items = []
    for k, i in enumerate(idx):
        e, other = bench[i], bench[idx[(k + 1) % len(idx)]]
        items.append(
            {
                "source": e.prefix_text,
                "systems": {
                    "ar": outs[i],
                    "lead1": e.prefix_text.split(" || ")[0].split(". ")[0] + ".",
                    "mismatch": other.target_text,
                    "human_ref": e.target_text,
                },
            }
        )
    json.dump(items, open(f"{prefix}_in.json", "w"))
    print("sample outputs:", *outs[:4], sep="\n  ")


if __name__ == "__main__":
    train() if sys.argv[1] == "train" else evaluate(int(sys.argv[2]), sys.argv[3])
