"""Trains the coarse-to-fine NextConceptTransformer against the frozen tokenizer.

No official training script exists upstream (see docs/constraints.md). This
reconstructs the teacher-forcing contract implied by `NextConceptTransformer.forward`
and `generate` (models/next_concept.py) and `MultiscaleResidualVectorQuantizer`
(models/quant.py), matching the released xsum checkpoint's architecture family
(raw_token_prefix + query_level0 + tie_heads_to_codebook):

- Input `x` to `forward()` is [prefix_embed; target_embed] in the *frozen vqvae's*
  embed_dim (forward applies its own Linear projection to n_embd internally) --
  NOT token ids.
- prefix_embed = vqvae.encoder.wte(prefix_token_ids) (raw_token_prefix convention,
  see next_concept.py's generate(), non-contextual branch).
- target_embed, for query_level0 models, is
  [net.build_l0_query_input(B); vqvae.vq.idx_to_ncm_input(idx)] -- the l0 query slot
  is a learned parameter (not data-dependent), and idx_to_ncm_input builds each
  subsequent level's input from the *cumulative* quantized reconstruction of all
  coarser levels (see quant.py:671-694). vqvae.get_ncm_inputs() only covers the
  non-query_level0 case, so this is the query_level0 extension of that helper.
- Loss is cross-entropy against the full per-level codebook indices `idx` (all
  levels, since query_level0=True predicts level 0 too -- vs the predict_level0
  path used by LM1B-style models, which is not used here).
"""

import torch
import torch.nn.functional as F
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from ncm_graphrag.data.map_stage import generate_examples
from ncm_graphrag.ncm.config import build_model_cfg, build_tokenizer_cfg

STEPS = 4000
BATCH_SIZE = 128
LR = 3e-4
N_EXAMPLES = 8000
SEED = 1


def _build_dataset(prefix_len: int, context_length: int, pre_tok) -> tuple[torch.Tensor, torch.Tensor]:
    examples = generate_examples(n=N_EXAMPLES, seed=SEED)
    pad_id, eos_id = pre_tok.pad_token_id, pre_tok.eos_token_id
    prefix_rows, target_rows = [], []
    for ex in examples:
        pids = pre_tok.encode(ex.prefix_text)
        pids = (pids[:prefix_len] + [pad_id] * prefix_len)[:prefix_len]

        tids = pre_tok.encode(ex.target_text) + [eos_id]
        if len(tids) > context_length:
            raise ValueError(f"target exceeds context_length={context_length}: {ex.target_text!r}")
        tids = tids + [pad_id] * (context_length - len(tids))

        prefix_rows.append(pids)
        target_rows.append(tids)
    return (
        torch.tensor(prefix_rows, dtype=torch.long),
        torch.tensor(target_rows, dtype=torch.long),
    )


def main() -> None:
    torch.manual_seed(SEED)
    tok_cfg = build_tokenizer_cfg()
    model_cfg = build_model_cfg()
    pre_tok = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vqvae = VQVAE(**tok_cfg.vqvae).to(device)
    state = torch.load(tok_cfg.save_path, map_location="cpu", weights_only=False)
    vqvae.load_state_dict(state["model"], strict=True)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    net = NextConceptTransformer(vqvae_local=vqvae, config=model_cfg).to(device)
    net.train()

    prefix_data, target_data = _build_dataset(model_cfg.prefix_len, tok_cfg.vqvae.context_length, pre_tok)
    n = prefix_data.shape[0]

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)

    for step in range(STEPS):
        sel = torch.randint(0, n, (BATCH_SIZE,))
        prefix_batch = prefix_data[sel].to(device)
        target_batch = target_data[sel].to(device)

        with torch.no_grad():
            idx = vqvae.encode(target_batch)  # (B, L) codebook indices across all levels
            rest_embed = vqvae.vq.idx_to_ncm_input(idx)  # (B, L - levels[0], embed_dim)
            z_prefix = vqvae.encoder.wte(prefix_batch)[:, : model_cfg.prefix_len, :]
        l0_embed = net.build_l0_query_input(BATCH_SIZE, dtype=rest_embed.dtype)
        x_target = torch.cat([l0_embed, rest_embed], dim=1)
        x = torch.cat([z_prefix, x_target], dim=1)

        logits = net(x)  # (B, L, max_vocab)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == STEPS - 1:
            acc = (logits.argmax(-1) == idx).float().mean().item()
            print(f"step {step:5d}  loss {loss.item():.4f}  level_acc {acc:.4f}", flush=True)

    import os

    os.makedirs(os.path.dirname(model_cfg.save_path), exist_ok=True)
    torch.save({"model": net.state_dict()}, model_cfg.save_path)
    print(f"saved {model_cfg.save_path}")


if __name__ == "__main__":
    main()
