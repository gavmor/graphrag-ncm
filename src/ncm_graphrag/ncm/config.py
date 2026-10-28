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


# --- Entity extraction task (second slice) ---
#
# Targets graphrag's real delimiter-tuple grammar (see src/ncm_graphrag/data/extraction.py
# and graphrag/index/operations/extract_graph/graph_extractor.py), not JSON. Scoped to
# entity-only output (not entity+relationship): a longer 160-token/6-level version was
# tried first and its free-running generation degraded badly (0/30 well-formed even
# with best-of-32) from coarse-to-fine exposure bias compounding over more codes -- see
# docs/constraints.md #6 and the extraction.py module docstring for the measured
# before/after. This config follows the Gemini background doc's own suggested fix:
# decouple into a smaller dedicated entity model.

EXTRACTION_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "extraction"
EXTRACTION_LEVELS = [2, 5, 10, 20, 80]
EXTRACTION_CONTEXT_LENGTH = 80  # bounded output length (gpt2 tokens); compact template runs ~55-68
EXTRACTION_PREFIX_LEN = 64  # bounded input length (gpt2 tokens) for "entity_types + text_unit"


def build_extraction_tokenizer_cfg() -> DictConfig:
    """VQVAE tokenizer config for the extraction task. `aux.fine_drop` is the nested
    codebook dropout the Gemini background doc (docs/NCM Training for GraphRAG.md ##3)
    describes -- it's already implemented upstream in models/quant.py:262-268 and used
    here the same way as the map-stage tokenizer, not a new mechanism built for this task."""
    cfg = OmegaConf.create(
        {
            "vqvae": {
                "in_vocab_size": GPT2_VOCAB_SIZE,
                "context_length": EXTRACTION_CONTEXT_LENGTH,
                "embed_dim": 64,
                "n_layers": 2,
                "vqvae_vocab_size": [48, 48, 48, 48, 48],
                "compression_factor": 1,
                "dropout": 0.1,
                "pre_quant_groupnorm": 8,
                "pre_quant_dropout": 0.1,
                "pre_quant_norm_type": "group",
                "use_qk_norm": True,
                "dec_layers": 1,
                "enc_layers": 2,
                "vector_quantizer_config": {
                    "clss": "multiscale_residual_vector_quantizer",
                    "decay": 0.99,
                    "epsilon": 1.0e-5,
                    "commitment_cost": 0.25,
                    "learned_l1_sampling": True,
                    "learned_all_sampling": False,
                    "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
                    "levels": {"use_manual_levels": True, "manual_levels": EXTRACTION_LEVELS},
                    "aux": {"fine_drop": {"prob": 0.3, "min_keep": 1}},
                },
            },
            "save_path": str(EXTRACTION_ARTIFACT_DIR / "tokenizer.pt"),
        }
    )
    assert cfg.vqvae.vector_quantizer_config.levels.manual_levels[-1] == cfg.vqvae.context_length
    assert len(cfg.vqvae.vqvae_vocab_size) == len(EXTRACTION_LEVELS)
    return cfg


def build_extraction_model_cfg() -> DictConfig:
    """NextConceptTransformer config for the extraction task:
    - Scaled capacity: 6 layers, 256 embedding dim (4.84M params, vs toy 0.84M)
    - Exposure bias mitigation: conditioning_noise=0.08 (docs/NCM Training for GraphRAG.md §5)
    - Compact 64-token target representation with 5 levels [2, 4, 8, 16, 64]
    """
    cfg = OmegaConf.create(
        {
            "n_layer": 6,
            "n_head": 6,
            "bias": True,
            "dropout": 0.1,
            "n_embd": 384,
            "vocab_size": [48, 48, 48, 48, 48],
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "prefix_len": EXTRACTION_PREFIX_LEN,
            "raw_token_prefix": True,
            "contextual_prefix": True,
            "prefix_pad_token_id": PAD_TOKEN_ID,
            "use_rope": True,
            "rope_base": 10000,
            "use_qk_norm": True,
            "query_level0": True,
            "tie_heads_to_codebook": True,
            "tie_heads_learn_scale": True,
            "conditioning_noise": 0.08,
            "save_path": str(EXTRACTION_ARTIFACT_DIR / "generator.pt"),
        }
    )
    assert len(cfg.vocab_size) == len(EXTRACTION_LEVELS)
    return cfg


def build_extraction_generation_cfg() -> DictConfig:
    """Decode-time settings. Unlike map-stage, frequency_penalty/adjacency_penalty are
    OFF here: measured directly (see docs/architecture.md) -- the extraction target
    legitimately repeats whole phrases across its two entity records (same template,
    "ORGANIZATION", "operates in <domain> and was named in the text unit."), and
    eval_shard.py's penalty decode loop, copied unmodified from the released xsum/wmt14
    configs (low-repetition natural-language targets), actively suppresses that correct
    repetition here: 0/30 well-formed with the penalty on (same recipe as map-stage) vs.
    28/30 greedy / 30/30 best-of-4 with it off. A genuinely from-scratch-tuned decode
    strategy, not a borrowed one, was necessary for this task -- see docs/constraints.md #6."""
    return OmegaConf.create(
        {
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.0,
            "cfg_scale": 0.0,
            "frequency_penalty": 0.0,
            "frequency_deficit": 0.0,
            "adjacency_penalty": "",
            "cut_at_eos": True,
        }
    )
