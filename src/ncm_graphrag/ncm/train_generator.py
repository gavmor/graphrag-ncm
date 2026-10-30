"""Trains the coarse-to-fine NextConceptTransformer against the frozen tokenizer, for
either task registered in src/ncm_graphrag/ncm/task_registry.py.

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

import argparse
import os

import torch
import torch.nn.functional as F
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from ncm_graphrag.ncm.task_registry import TASKS, TaskSpec

# Schedule knobs are overridable so schedule experiments need no code edits. The sample budget
# (batch * steps) is what bounds wall-clock, so trading one against the other re-optimises in the
# same time. Defaults reproduce the shipped recipe exactly.
BATCH_SIZE = int(os.environ.get("NCM_BATCH_SIZE", "128"))
LR = float(os.environ.get("NCM_LR", "3e-4"))
COND_DROP = float(os.environ.get("NCM_COND_DROP", "0"))
TRAIN_PAD_MASK = os.environ.get("NCM_TRAIN_PAD_MASK", "0") == "1" or COND_DROP > 0.0
SEED = int(os.environ.get("NCM_SEED", "1"))


def apply_cond_drop(prefix: torch.Tensor, p: float, pad_id: int, eos_id: int) -> torch.Tensor:
    """Per-item condition dropout for classifier-free guidance.

    With probability `p` per row, replace the whole prefix with the exact null prefix that
    NextConceptTransformer.generate() builds for CFG: [EOS, PAD, PAD, ...]. Targets and the loss
    are untouched, so gradient scale does not depend on how many rows drop. `p=1.0` drops every row.
    """
    drop = torch.rand(prefix.shape[0], 1, device=prefix.device) < p
    null_row = torch.full_like(prefix[:1], pad_id)
    null_row[:, 0] = eos_id
    return torch.where(drop, null_row.expand_as(prefix), prefix)


def _build_dataset(task: TaskSpec, prefix_len: int, context_length: int, pre_tok) -> tuple[torch.Tensor, torch.Tensor]:
    examples = task.generate_examples(task.generator_n_examples, SEED)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="mapstage")
    args = ap.parse_args()
    task = TASKS[args.task]

    torch.manual_seed(SEED)
    tok_cfg = task.build_tokenizer_cfg()
    model_cfg = task.build_model_cfg()
    pre_tok = HFTokenizer(model_id="gpt2", pad_token="<|pad|>")
    pad_id, eos_id = pre_tok.pad_token_id, pre_tok.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vqvae = VQVAE(**tok_cfg.vqvae).to(device)
    state = torch.load(tok_cfg.save_path, map_location="cpu", weights_only=False)
    vqvae.load_state_dict(state["model"], strict=True)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    net = NextConceptTransformer(vqvae_local=vqvae, config=model_cfg).to(device)
    net.train()

    prefix_data, target_data = _build_dataset(task, model_cfg.prefix_len, tok_cfg.vqvae.context_length, pre_tok)
    n = prefix_data.shape[0]

    # Optional held-out teacher-forced loss every NCM_EVAL_EVERY steps (off by default). Training loss
    # alone cannot show memorisation: on the Multi-News drafter it fell to ~1.8 while held-out loss
    # was ~3.9. Uses the task's benchmark examples (never trained on) and the same encoding path.
    eval_every = int(os.environ.get("NCM_EVAL_EVERY", "0"))
    heldout = None
    if eval_every and task.benchmark_examples is not None:
        import dataclasses

        held = task.benchmark_examples()[:256]
        fixed = dataclasses.replace(task, generate_examples=lambda n, s: held, generator_n_examples=len(held))
        h_pre, h_tgt = _build_dataset(fixed, model_cfg.prefix_len, tok_cfg.vqvae.context_length, pre_tok)
        heldout = (h_pre.to(device), h_tgt.to(device))

    @torch.no_grad()
    def heldout_loss(h_pre: torch.Tensor, h_tgt: torch.Tensor) -> tuple[float, float]:
        net.eval()
        total = correct = count = 0.0
        for i in range(0, len(h_tgt), 64):
            p, t = h_pre[i : i + 64], h_tgt[i : i + 64]
            h_idx = vqvae.encode(t)
            h_rest = vqvae.vq.idx_to_ncm_input(h_idx)
            h_enc = vqvae.encoder
            if bool(getattr(model_cfg, "contextual_prefix", False)) and len(getattr(h_enc, "blocks", [])) > 0:
                h_z = h_enc.forward_nopos(p)[:, : model_cfg.prefix_len, :]
            else:
                h_z = h_enc.wte(p)[:, : model_cfg.prefix_len, :]
            h_x = torch.cat([h_z, torch.cat([net.build_l0_query_input(len(p), dtype=h_rest.dtype), h_rest], 1)], 1)
            h_mask = (p[:, : model_cfg.prefix_len] == pad_id) if TRAIN_PAD_MASK else None
            h_logits = net(h_x, prefix_pad_mask=h_mask)
            h_ce = F.cross_entropy(h_logits.reshape(-1, h_logits.shape[-1]), h_idx.reshape(-1), reduction="sum")
            total += h_ce.item()
            correct += (h_logits.argmax(-1) == h_idx).float().sum().item()
            count += h_idx.numel()
        net.train()
        return total / count, correct / count

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)

    steps = int(os.environ.get("NCM_STEPS", str(task.generator_steps)))
    for step in range(steps):
        sel = torch.randint(0, n, (BATCH_SIZE,))
        prefix_batch = prefix_data[sel].to(device)
        target_batch = target_data[sel].to(device)
        if COND_DROP > 0.0:
            prefix_batch = apply_cond_drop(prefix_batch, COND_DROP, pad_id, eos_id)

        with torch.no_grad():
            idx = vqvae.encode(target_batch)  # (B, L) codebook indices across all levels
            rest_embed = vqvae.vq.idx_to_ncm_input(idx)  # (B, L - levels[0], embed_dim)
            _enc = vqvae.encoder
            if bool(getattr(model_cfg, "contextual_prefix", False)) and len(getattr(_enc, "blocks", [])) > 0:
                # Deliberately unmasked: measured PAD-masked prefix attention at 0.89 F1 vs
                # 1.83 with padding attended (the PAD slots act as attention sinks), so the
                # upstream helper is kept as-is.
                z_prefix = _enc.forward_nopos(prefix_batch)[:, : model_cfg.prefix_len, :]
            else:
                z_prefix = _enc.wte(prefix_batch)[:, : model_cfg.prefix_len, :]
            if hasattr(model_cfg, "conditioning_noise") and float(model_cfg.conditioning_noise) > 0:
                c_noise = float(model_cfg.conditioning_noise)
                mask = (torch.rand(BATCH_SIZE, 1, 1, device=device) < 0.4).float()
                rest_embed = rest_embed + mask * (c_noise * torch.randn_like(rest_embed))
        l0_embed = net.build_l0_query_input(BATCH_SIZE, dtype=rest_embed.dtype)
        x_target = torch.cat([l0_embed, rest_embed], dim=1)
        x = torch.cat([z_prefix, x_target], dim=1)

        # Mask pad keys exactly as generate() does (-inf where the prefix token == pad). Off by
        # default so existing recipes stay reproducible; forced on with condition dropout, because
        # an unmasked null row would show 287 pad tokens at train time but one EOS at decode time.
        pad_mask = (prefix_batch[:, : model_cfg.prefix_len] == pad_id) if TRAIN_PAD_MASK else None
        logits = net(x, prefix_pad_mask=pad_mask)  # (B, L, max_vocab)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == steps - 1:
            acc = (logits.argmax(-1) == idx).float().mean().item()
            print(f"[{args.task}] step {step:5d}  loss {loss.item():.4f}  level_acc {acc:.4f}", flush=True)

        if eval_every and (step % eval_every == 0 or step == steps - 1) and heldout is not None:
            h_loss, h_acc = heldout_loss(*heldout)
            print(f"[{args.task}] step {step:5d}  HELDOUT loss {h_loss:.4f}  acc {h_acc:.4f}", flush=True)

    os.makedirs(os.path.dirname(model_cfg.save_path), exist_ok=True)
    torch.save({"model": net.state_dict()}, model_cfg.save_path)
    print(f"saved {model_cfg.save_path}")


if __name__ == "__main__":
    main()
