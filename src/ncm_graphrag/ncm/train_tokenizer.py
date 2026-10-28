"""Trains the multiscale residual-VQ tokenizer on map-stage target JSON strings.

No official training script exists upstream for this repo (see docs/constraints.md) --
this reimplements the standard VQ-VAE-with-EMA-codebook recipe directly against
models.vqvae.VQVAE: reconstruction cross-entropy + the commitment/codebook loss the
module already computes in `forward()`. The codebook itself is EMA-updated inside
`MultiscaleResidualVectorQuantizer.forward()` when `self.training` is True (see
models/quant.py:485-500) -- the optimizer only needs to step the encoder/decoder/
quant_resi parameters (codebook embeddings have `requires_grad=False`).
"""

import torch
import torch.nn.functional as F
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from ncm_graphrag.data.map_stage import generate_examples
from ncm_graphrag.ncm.config import build_tokenizer_cfg

STEPS = 3000
BATCH_SIZE = 128
LR = 3e-4
N_EXAMPLES = 6000
SEED = 0


def _build_dataset(cfg, pre_tok) -> torch.Tensor:
    examples = generate_examples(n=N_EXAMPLES, seed=SEED)
    context_length = cfg.vqvae.context_length
    rows = []
    for ex in examples:
        ids = pre_tok.encode(ex.target_text) + [pre_tok.eos_token_id]
        if len(ids) > context_length:
            raise ValueError(f"target exceeds context_length={context_length} ({len(ids)} tokens): {ex.target_text!r}")
        ids = ids + [pre_tok.pad_token_id] * (context_length - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def main() -> None:
    torch.manual_seed(SEED)
    cfg = build_tokenizer_cfg()
    pre_tok = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")

    data = _build_dataset(cfg, pre_tok)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqvae = VQVAE(**cfg.vqvae).to(device)
    vqvae.train()

    opt = torch.optim.AdamW(vqvae.parameters(), lr=LR, weight_decay=0.01)
    n = data.shape[0]

    for step in range(STEPS):
        batch_idx = torch.randint(0, n, (BATCH_SIZE,))
        batch = data[batch_idx].to(device)

        x_recon, vq_loss, _per_level_util, global_util, _ppl = vqvae(batch)
        recon_loss = F.cross_entropy(x_recon.reshape(-1, x_recon.shape[-1]), batch.reshape(-1))
        loss = recon_loss + vq_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == STEPS - 1:
            acc = (x_recon.argmax(-1) == batch).float().mean().item()
            print(
                f"step {step:5d}  loss {loss.item():.4f}  recon {recon_loss.item():.4f}  "
                f"vq {vq_loss.item():.4f}  token_acc {acc:.4f}  codebook_util {global_util:.3f}",
                flush=True,
            )

    cfg.save_path and __import__("os").makedirs(__import__("os").path.dirname(cfg.save_path), exist_ok=True)
    torch.save({"model": vqvae.state_dict()}, cfg.save_path)
    print(f"saved {cfg.save_path}")


if __name__ == "__main__":
    main()
