"""Config for a tiny, from-scratch-trained NCM tokenizer+generator pair sized for
GraphRAG's Global Search map-stage task: short "community report + query" prefix in,
bounded `{"points":[{"description":...,"score":...}]}` JSON out.

Shapes mirror the released xsum checkpoint's architecture choices (raw_token_prefix,
query_level0, tie_heads_to_codebook, block_diagonal attention, RoPE) at a fraction of
the size, since no official training recipe/hyperparameters are released upstream
(see docs/constraints.md) -- these values were chosen to fit the task's short bounded
output and are not tuned beyond "trains and produces non-degenerate text" (see
docs/architecture.md for the smoke-test results).
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "mapstage"

GPT2_VOCAB_SIZE = 50304  # rounded up from gpt2 (50257) + 1 added pad token (50258), matches xsum.yaml's convention
PAD_TOKEN_ID = 50257
EOS_TOKEN_ID = 50256

# Coarse-to-fine level lengths for the multiscale residual VQ; last level == vqvae.context_length.
LEVELS = [2, 4, 8, 32]
CONTEXT_LENGTH = 32  # bounded output length (gpt2 tokens) for the target JSON
PREFIX_LEN = 96  # bounded input length (gpt2 tokens) for "community report + query"


def build_tokenizer_cfg() -> DictConfig:
    """VQVAE (multiscale residual VQ tokenizer) config, trained standalone and frozen."""
    cfg = OmegaConf.create(
        {
            "vqvae": {
                "in_vocab_size": GPT2_VOCAB_SIZE,
                "context_length": CONTEXT_LENGTH,
                "embed_dim": 64,
                "n_layers": 2,  # attention head count inside Encoder/Decoder blocks, not depth
                "vqvae_vocab_size": [48, 48, 48, 48],
                "compression_factor": 1,
                "dropout": 0.1,
                "pre_quant_groupnorm": 8,
                "pre_quant_dropout": 0.1,
                "pre_quant_norm_type": "group",
                "use_qk_norm": True,
                "dec_layers": 1,
                "vector_quantizer_config": {
                    "clss": "multiscale_residual_vector_quantizer",
                    "decay": 0.99,
                    "epsilon": 1.0e-5,
                    "commitment_cost": 0.25,
                    "learned_l1_sampling": True,
                    "learned_all_sampling": False,
                    "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
                    "levels": {"use_manual_levels": True, "manual_levels": LEVELS},
                    "aux": {"fine_drop": {"prob": 0.3, "min_keep": 1}},
                },
            },
            "save_path": str(ARTIFACT_DIR / "tokenizer.pt"),
        }
    )
    assert cfg.vqvae.vector_quantizer_config.levels.manual_levels[-1] == cfg.vqvae.context_length
    assert len(cfg.vqvae.vqvae_vocab_size) == len(LEVELS)
    return cfg


def build_model_cfg() -> DictConfig:
    """NextConceptTransformer (coarse-to-fine generator) config, same architecture family
    as the released xsum checkpoint (raw_token_prefix + query_level0 + tie_heads_to_codebook)."""
    cfg = OmegaConf.create(
        {
            "n_layer": 4,
            "n_head": 4,
            "bias": True,
            "dropout": 0.1,
            "n_embd": 128,
            "vocab_size": [48, 48, 48, 48],  # must equal vqvae_vocab_size (query_level0=True predicts all levels)
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "prefix_len": PREFIX_LEN,
            "raw_token_prefix": True,
            "prefix_pad_token_id": PAD_TOKEN_ID,
            "use_rope": True,
            "rope_base": 10000,
            "use_qk_norm": True,
            "query_level0": True,
            "tie_heads_to_codebook": True,
            "tie_heads_learn_scale": True,
            "save_path": str(ARTIFACT_DIR / "generator.pt"),
        }
    )
    assert len(cfg.vocab_size) == len(LEVELS)
    return cfg


def build_generation_cfg() -> DictConfig:
    """Decode-time settings. Greedy argmax is degenerate/repetitive (see docs/constraints.md);
    frequency_penalty + adjacency_penalty (copied from eval_shard.py's pick_tokens) is required
    for non-garbage output, same as the released xsum/wmt14 configs use."""
    return OmegaConf.create(
        {
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.0,
            "cfg_scale": 0.0,
            "frequency_penalty": 1.5,
            "frequency_deficit": 0.0,
            "adjacency_penalty": "3,3,0.6",
            "cut_at_eos": True,
        }
    )
