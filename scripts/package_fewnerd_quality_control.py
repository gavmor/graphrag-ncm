"""Package approved control source and cached GPT-2 assets without credentials."""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_REVISION = "22fd687e06f83b7aea8ccb466570b2197905d1b2"
TEMPLATE_BLOB = "066caf22ff0e12b18a6c25f0797689dbf060eb9b"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def package(output: Path, image_digest: str):
    tracked = subprocess.check_output(["git", "ls-files", "--recurse-submodules", "-z"], cwd=ROOT).decode().split("\0")
    directories = ("src/", "scripts/", "tests/", "docs/", "concourse/", "benchmarks/", "third_party/ncm/")
    standalone = {"autoresearch.sh", "pyproject.toml", "uv.lock", "README.md"}
    selected = sorted(name for name in tracked if name and (name in standalone or name.startswith(directories)))
    payloads = {}
    for name in selected:
        path = ROOT / name
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Source file is missing or escapes the source tree: {name}")
        if set(path.parts) & {".git", ".secrets", ".venv", "__pycache__"}:
            raise ValueError(f"Forbidden source path: {name}")
        payloads["repo/" + name] = path.read_bytes()
    if "repo/benchmarks/fewnerd-protected/frozen.json" not in payloads:
        raise ValueError("Protected protocol must be committed before packaging")
    cache = f"hf-cache/hub/models--gpt2/snapshots/{TOKENIZER_REVISION}"
    for path in sorted((ROOT / "benchmarks/fewnerd/tokenizer").iterdir()):
        payloads[f"{cache}/{path.name}"] = path.read_bytes()
    payloads["hf-cache/hub/models--gpt2/refs/main"] = TOKENIZER_REVISION.encode()
    info = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "ncm_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/ncm")
        .decode()
        .strip(),
        "image_digest": image_digest,
        "template_revision": TEMPLATE_REVISION,
        "template_blob": TEMPLATE_BLOB,
        "cuda_extensions": "Uses installed pinned Torch binaries; no new AOT extensions required",
        "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
    }
    payloads["source-info.json"] = (json.dumps(info, indent=2, sort_keys=True) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output, suffix=".tmp") as temporary:
        with gzip.GzipFile(fileobj=temporary, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, content in sorted(payloads.items()):
                    entry = tarfile.TarInfo(name)
                    entry.size = len(content)
                    entry.mode = 0o755 if name.endswith(".sh") else 0o644
                    entry.mtime = 0
                    archive.addfile(entry, io.BytesIO(content))
        temporary.flush()
        digest = hashlib.sha256(Path(temporary.name).read_bytes()).hexdigest()
        version = "r1-" + digest[:16]
        target = output / f"fewnerd-quality-inputs-{version}.tar.gz"
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"Refusing to replace different immutable input: {target}")
        if not target.exists():
            target.write_bytes(Path(temporary.name).read_bytes())
    result = {
        "archive": str(target),
        "sha256": digest,
        "input_version": version,
        "source_revision": info["source_revision"],
        "files": len(payloads),
    }
    (output / "bundle.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    args = parser.parse_args()
    package(args.output, args.image_digest)
