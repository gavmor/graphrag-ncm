"""Does the generator use its source prefix? Teacher-forced loss with correct vs shuffled vs null prefix.

    NCM_CORPORA_ROOT=... [SPLIT=train|val] [TASK=report_multinews] uv run python scripts/prefix_use.py

Interpretation: if the correct-prefix loss is not clearly below the shuffled/null loss on HELD-OUT data,
the model is not using its source there. Compare SPLIT=train against SPLIT=val: a large gap between
the two is memorisation. Free-running ROUGE cannot show this (it failed to separate them here).
Uses the raw-token prefix path only (contextual_prefix=False).
"""

import dataclasses
import importlib
import os

import torch
import torch.nn.functional as F

# Side-effect import that puts the vendored ncm repo on sys.path. A statement (not an import line)
# so import sorting cannot move it below the imports that need it.
importlib.import_module("ncm_graphrag.ncm._ncm_repo")

from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402
from tokenization.hf_tokenizer import HFTokenizer  # noqa: E402

from ncm_graphrag.ncm.task_registry import TASKS  # noqa: E402
from ncm_graphrag.ncm.train_generator import _build_dataset  # noqa: E402

N = 384
spec = TASKS[os.environ.get("TASK", "report_multinews")]
tc, mc = spec.build_tokenizer_cfg(), spec.build_model_cfg()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
vq = VQVAE(**tc.vqvae).to(dev)
vq.load_state_dict(torch.load(tc.save_path, map_location="cpu", weights_only=False)["model"])
vq.eval()
net = NextConceptTransformer(vqvae_local=vq, config=mc).to(dev)
net.load_state_dict(torch.load(mc.save_path, map_location="cpu", weights_only=False)["model"])
net.eval()

split = os.environ.get("SPLIT", "val")
if split == "train":
    examples = spec.generate_examples(spec.generator_n_examples, 1)[:N]
else:
    assert spec.benchmark_examples is not None, "task has no held-out benchmark examples"
    examples = spec.benchmark_examples()[:N]
fixed = dataclasses.replace(spec, generate_examples=lambda n, s: examples, generator_n_examples=len(examples))
pre, tgt = _build_dataset(fixed, mc.prefix_len, tc.vqvae.context_length, tok)
pre, tgt = pre.to(dev), tgt.to(dev)


@torch.no_grad()
def teacher_forced(prefix: torch.Tensor) -> tuple[float, float]:
    total = correct = count = 0.0
    for i in range(0, len(tgt), 64):
        p, t = prefix[i : i + 64], tgt[i : i + 64]
        idx = vq.encode(t)
        rest = vq.vq.idx_to_ncm_input(idx)
        z = vq.encoder.wte(p)[:, : mc.prefix_len, :]
        x = torch.cat([z, torch.cat([net.build_l0_query_input(len(p), dtype=rest.dtype), rest], 1)], 1)
        logits = net(x)
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1), reduction="sum").item()
        correct += (logits.argmax(-1) == idx).float().sum().item()
        count += idx.numel()
    return total / count, correct / count


perm = torch.randperm(len(pre), generator=torch.Generator().manual_seed(0)).to(dev)
null = torch.full_like(pre, tok.pad_token_id)
null[:, 0] = tok.eos_token_id
for name, prefix in (("correct", pre), ("shuffled", pre[perm]), ("null", null)):
    loss, acc = teacher_forced(prefix)
    print(f"[{split}] {name:<9} teacher-forced loss {loss:.4f}  acc {acc:.4f}")
