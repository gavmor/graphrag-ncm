"""Side-effect import: puts the vendored HazyResearch/ncm submodule on sys.path.

The ncm repo (third_party/ncm) is a flat research codebase, not an installed
package -- its own scripts do `sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))`
before `from models.vqvae import VQVAE` etc. We replicate that convention so our
training/serving code can import `models.*`, `tokenization.*`, `utils.*` directly
from the submodule without vendoring or packaging it ourselves.

Usage: `from ncm_graphrag.ncm import _ncm_repo  # noqa: F401` before importing
anything from `models`/`tokenization`/`utils`.
"""

import os
import sys
from pathlib import Path

NCM_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "ncm"

if not (NCM_ROOT / "models").is_dir():
    raise RuntimeError(
        f"third_party/ncm submodule not found at {NCM_ROOT} -- run `git submodule update --init --recursive`."
    )

if str(NCM_ROOT) not in sys.path:
    sys.path.insert(0, str(NCM_ROOT))

os.environ.setdefault("NCM_ROOT", str(NCM_ROOT))
