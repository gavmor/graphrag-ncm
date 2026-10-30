"""Train and compare REBEL graph extractors with independent held-out evaluation.

Run `prepare`, `tokenizer`, `ncm`, `ar`, then `evaluate --model ncm` and
`evaluate --model ar`. All checkpoints and encoded data live under artifacts/.
Gold transfer sets and the second half of the REBEL validation sample are never
used for checkpoint selection. The default runs are deliberately configurable;
record actual steps and held-out metrics rather than assuming paper-scale training.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm_graphrag.data import rebel_extraction as data
from ncm_graphrag.ncm import config

importlib.import_module("ncm_graphrag.ncm._ncm_repo")

from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402
from tokenization.hf_tokenizer import HFTokenizer  # noqa: E402

ROOT = config.REBEL_ARTIFACT_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = config.GPT2_VOCAB_SIZE
WIDTH, LAYERS, HEADS = 768, 13, 12
PREFIX = data.PREFIX_TOKEN_BUDGET
TARGET = data.TARGET_TOKEN_BUDGET
MAXLEN = PREFIX + TARGET + 1


def tokenizer():
    return HFTokenizer(model_id="gpt2", pad_token="<|pad|>")


def encode_examples(examples, tok):
    prefixes = torch.full((len(examples), PREFIX), tok.pad_token_id, dtype=torch.int32)
    targets = torch.full((len(examples), TARGET), tok.pad_token_id, dtype=torch.int32)
    lengths = torch.empty((len(examples), 2), dtype=torch.int32)
    for i, example in enumerate(examples):
        source = tok.encode(example.prefix_text)
        target = tok.encode(example.target_text) + [tok.eos_token_id]
        if len(source) > PREFIX or len(target) > TARGET:
            raise ValueError(f"example {example.source_id!r} exceeds its token budget")
        prefixes[i, : len(source)] = torch.tensor(source, dtype=torch.int32)
        targets[i, : len(target)] = torch.tensor(target, dtype=torch.int32)
        lengths[i] = torch.tensor([len(source), len(target)], dtype=torch.int32)
    return {"prefix": prefixes, "target": targets, "lengths": lengths}


def prepare(args):
    ROOT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    start = time.monotonic()
    examples = data._split_examples("train", args.train_examples, 1)
    encoded = encode_examples(examples, tok)
    encoded["source_ids"] = [e.source_id for e in examples]
    torch.save(encoded, ROOT / f"train_{len(examples)}.pt")
    print(f"prepared train: {len(examples)} examples in {time.monotonic() - start:.1f}s", flush=True)

    start = time.monotonic()
    selected, heldout = data.validation_partitions(args.validation_examples)
    examples = selected + heldout
    encoded = encode_examples(examples, tok)
    encoded["source_ids"] = [e.source_id for e in examples]
    torch.save(encoded, ROOT / f"validation_{len(examples)}.pt")
    print(f"prepared validation: {len(examples)} examples in {time.monotonic() - start:.1f}s", flush=True)


def load_encoded(name, n):
    return torch.load(ROOT / f"{name}_{n}.pt", map_location="cpu", weights_only=True)


def checkpoint(path, model, step, loss, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step, "validation_loss": loss, **(extra or {})}, path)


def optimizer_step(loss, optimizer, scaler, model):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_tokenizer(args):
    torch.manual_seed(args.seed)
    train = load_encoded("train", args.train_examples)["target"]
    validation = load_encoded("validation", args.validation_examples * 2)["target"][: args.validation_examples]
    cfg = config.build_rebel_tokenizer_cfg()
    model = VQVAE(**cfg.vqvae).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best = math.inf
    for step in range(1, args.steps + 1):
        model.train()
        ids = train[torch.randint(len(train), (args.batch_size,))].long().to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits, vq_loss, *_ = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1)) + vq_loss
        optimizer_step(loss, optimizer, scaler, model)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            total, count, correct = 0.0, 0, 0
            with torch.no_grad():
                for start in range(0, len(validation), args.batch_size):
                    batch = validation[start : start + args.batch_size].long().to(DEVICE)
                    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                        logits, _, *_ = model(batch)
                        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.reshape(-1), reduction="sum")
                    total += ce.item()
                    valid = batch != config.PAD_TOKEN_ID
                    correct += ((logits.argmax(-1) == batch) & valid).sum().item()
                    count += valid.sum().item()
            val_loss = total / validation.numel()
            if val_loss < best:
                best = val_loss
                checkpoint(Path(cfg.save_path), model, step, best, {"nonpad_accuracy": correct / max(count, 1)})
            print(
                f"tokenizer step={step} train={loss.item():.4f} val={val_loss:.4f} "
                f"nonpad_accuracy={correct / max(count, 1):.4f} best={best:.4f}",
                flush=True,
            )


def load_vq():
    cfg = config.build_rebel_tokenizer_cfg()
    model = VQVAE(**cfg.vqvae).to(DEVICE)
    model.load_state_dict(torch.load(cfg.save_path, map_location="cpu", weights_only=False)["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def ncm_loss(model, vq, prefix, target):
    with torch.no_grad():
        idx = vq.encode(target)
        rest = vq.vq.idx_to_ncm_input(idx)
        prefix_embed = vq.encoder.wte(prefix)
        query = model.build_l0_query_input(len(prefix), dtype=rest.dtype)
        x = torch.cat((prefix_embed, query, rest), dim=1)
    logits = model(x, prefix_pad_mask=prefix == config.PAD_TOKEN_ID)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1))


class ARBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(WIDTH)
        self.ln2 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.fc = nn.Linear(WIDTH, 4 * WIDTH)
        self.out = nn.Linear(4 * WIDTH, WIDTH)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        batch, length, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(WIDTH, dim=-1)
        q, k, v = (z.view(batch, length, HEADS, WIDTH // HEADS).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.1 if self.training else 0.0)
        x = x + self.dropout(self.proj(y.transpose(1, 2).reshape(batch, length, WIDTH)))
        return x + self.dropout(self.out(F.gelu(self.fc(self.ln2(x)))))


class ARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, WIDTH)
        self.wpe = nn.Embedding(MAXLEN, WIDTH)
        self.blocks = nn.ModuleList(ARBlock() for _ in range(LAYERS))
        self.ln = nn.LayerNorm(WIDTH)
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)

    def forward(self, ids, select=None):  # ty: ignore[invalid-mutable-override]
        x = self.wte(ids) + self.wpe(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if select is not None:
            x = x[select]
        return x @ self.wte.weight.T


def ar_batch(cache, indices, eos):
    lengths = cache["lengths"][indices]
    seqs = []
    source_lengths = []
    for row, (src_len, tgt_len) in zip(indices.tolist(), lengths.tolist(), strict=True):
        source = cache["prefix"][row, :src_len].long()
        target = cache["target"][row, :tgt_len].long()
        seqs.append(torch.cat((source, torch.tensor([eos]), target)))
        source_lengths.append(src_len + 1)
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), config.PAD_TOKEN_ID, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, (seq, start) in enumerate(zip(seqs, source_lengths, strict=True)):
        ids[i, : len(seq)] = seq
        labels[i, start : len(seq)] = seq[start:]
    return ids.to(DEVICE), labels.to(DEVICE)


def ar_loss(model, cache, indices, eos):
    ids, labels = ar_batch(cache, indices, eos)
    target = labels[:, 1:]
    logits = model(ids[:, :-1], target != -100)
    return F.cross_entropy(logits, target[target != -100])


def train_generator(args, kind):
    torch.manual_seed(args.seed)
    train = load_encoded("train", args.train_examples)
    valid = load_encoded("validation", args.validation_examples * 2)
    if kind == "ncm":
        vq = load_vq()
        model_cfg = config.build_rebel_model_cfg()
        model = NextConceptTransformer(vqvae_local=vq, config=model_cfg).to(DEVICE)
        path = Path(model_cfg.save_path)
    else:
        vq = None
        model = ARModel().to(DEVICE)
        path = ROOT / "ar_generator.pt"
    total = sum(p.numel() for p in model.parameters())
    non_embedding = total
    if kind == "ar":
        non_embedding -= VOCAB * WIDTH + MAXLEN * WIDTH
    print(f"{kind} total_parameters={total} non_embedding_parameters={non_embedding}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    eos = tokenizer().eos_token_id
    best = math.inf
    stale = 0
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(len(train["target"]), (args.batch_size,))
        lr = args.lr * min(1.0, step / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            if kind == "ncm":
                p = train["prefix"][ids].long().to(DEVICE)
                t = train["target"][ids].long().to(DEVICE)
                loss = ncm_loss(model, vq, p, t)
            else:
                loss = ar_loss(model, train, ids, eos)
        optimizer_step(loss, optimizer, scaler, model)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            total_loss, count = 0.0, 0
            with torch.no_grad():
                for start in range(0, args.validation_examples, args.batch_size):
                    indices = torch.arange(start, min(start + args.batch_size, args.validation_examples))
                    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                        if kind == "ncm":
                            p = valid["prefix"][indices].long().to(DEVICE)
                            t = valid["target"][indices].long().to(DEVICE)
                            val = ncm_loss(model, vq, p, t)
                        else:
                            val = ar_loss(model, valid, indices, eos)
                    total_loss += val.item() * len(indices)
                    count += len(indices)
            val_loss = total_loss / count
            if val_loss < best:
                best, stale = val_loss, 0
                checkpoint(path, model, step, best, {"train_examples": args.train_examples})
            else:
                stale += 1
            print(f"{kind} step={step} train={loss.item():.4f} val={val_loss:.4f} best={best:.4f}", flush=True)
            if stale >= args.patience:
                print(f"{kind} early_stop={step}", flush=True)
                break


@torch.no_grad()
def ar_generate(model, tok, text):
    source = tok.encode(text)[:PREFIX] + [tok.eos_token_id]
    ids = torch.tensor([source], dtype=torch.long, device=DEVICE)
    for _ in range(TARGET):
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            select = torch.zeros_like(ids, dtype=torch.bool)
            select[:, -1] = True
            next_token = int(model(ids, select)[0].argmax().item())
        if next_token == tok.eos_token_id:
            break
        ids = torch.cat((ids, torch.tensor([[next_token]], device=DEVICE)), dim=1)
    return tok.decode(ids[0, len(source) :].tolist(), skip_special_tokens=True).strip()


def evaluate(args):
    tok = tokenizer()
    if args.model == "ncm":
        from ncm_graphrag.ncm.generate import NCMTaskGenerator

        generator = NCMTaskGenerator(task="rebel_extraction", compile_blocks=False)
        generate = generator.generate
        state = torch.load(config.build_rebel_model_cfg().save_path, map_location="cpu", weights_only=False)
    else:
        model = ARModel().to(DEVICE)
        state = torch.load(ROOT / "ar_generator.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        def generate(text):
            return ar_generate(model, tok, text)

    _, val = data.validation_partitions(args.validation_examples)
    if args.eval_examples > len(val):
        raise ValueError("eval-examples cannot exceed the held-out REBEL validation partition")
    datasets = {"rebel_validation": val[: args.eval_examples]}
    coverage = {}
    for name in ("re_docred", "docred"):
        pool, coverage[name] = data.gold_pool(name, "dev")
        random.Random(7).shuffle(pool)
        if len(pool) < args.eval_examples:
            raise ValueError(f"not enough eligible {name} gold documents")
        datasets[f"{name}_dev"] = pool[: args.eval_examples]
    results = {"model": args.model, "checkpoint_step": state["step"], "checkpoint_val_loss": state["validation_loss"]}
    results["gold_split_coverage"] = coverage
    for dataset_name, examples in datasets.items():
        counts = [[0, 0, 0] for _ in range(3)]
        valid = 0
        samples = []
        for example in examples:
            pred = generate(example.prefix_text)
            valid += int(data.is_well_formed(pred))
            for count, gold, actual in zip(
                counts, data.graph_sets(example.target_text), data.graph_sets(pred), strict=True
            ):
                count[0] += len(gold & actual)
                count[1] += len(actual)
                count[2] += len(gold)
            if len(samples) < 3:
                samples.append(
                    {"id": example.source_id, "source": example.prefix_text, "gold": example.target_text, "pred": pred}
                )
        metrics = {}
        for name, (tp, pred, gold) in zip(("entity", "typed_entity", "relation"), counts, strict=True):
            precision = tp / pred if pred else 0.0
            recall = tp / gold if gold else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics[name] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "pred": pred, "gold": gold}
        metrics["valid_rate"] = valid / len(examples)
        metrics["examples"] = len(examples)
        metrics["samples"] = samples
        results[dataset_name] = metrics
        print(dataset_name, json.dumps({k: v for k, v in metrics.items() if k != "samples"}), flush=True)
    path = ROOT / f"{args.model}_evaluation.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "tokenizer", "ncm", "ar", "evaluate"))
    parser.add_argument("--train-examples", type=int, default=100000)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", choices=("ncm", "ar"), default="ncm")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "tokenizer":
        train_tokenizer(args)
    elif args.action in ("ncm", "ar"):
        train_generator(args, args.action)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
