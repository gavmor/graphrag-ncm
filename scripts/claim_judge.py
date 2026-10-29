"""Blind claim-level faithfulness judge (the decision-making metric for the drafter).

Usage (project venv; needs ANTHROPIC_API_KEY, optionally ANTHROPIC_WORKSPACE_ID):
    uv run python scripts/claim_judge.py INPUT.json OUTPUT.json

INPUT.json: list of {"source": str, "systems": {"<name>": "<summary>", ...}}.
Every (source, summary) pair is judged independently, in shuffled order, with no system names shown.
Reports, per system: share of claims supported, share of fully faithful summaries (bootstrap 95% CI),
mean fluency. Always include the two calibration anchors as systems:
  * "lead1"    - verbatim first sentence of the source (expect ~100% supported)
  * "mismatch" - a summary written for a DIFFERENT source (expect ~0% supported)
If the anchors miss those targets the judge prompt is mis-calibrated and the numbers are void.
"""

import collections
import json
import os
import random
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

import litellm

MODEL = "anthropic/claude-haiku-5-5"
PROMPT = (
    "You verify whether a one-sentence summary is faithful to a source text. Use ONLY the source: do not use "
    "world knowledge, and assume a claim is NOT supported unless the source explicitly states it or it follows "
    "directly from what the source states. Split the summary into its atomic factual claims. For each claim answer "
    "supported, unsupported (not stated in the source) or contradicted. Any name, number, place, date or "
    "organisation that does not appear in the source makes its claim unsupported. Also rate fluency 1-5 "
    "(5 = natural grammatical English, 1 = garbled). Reply with ONLY JSON: "
    '{"claims":[{"claim":"...","verdict":"supported|unsupported|contradicted"}],"fluency":<1-5>}'
)
_WS = os.environ.get("ANTHROPIC_WORKSPACE_ID")
HDR = {"anthropic-workspace-id": _WS} if _WS else {}
ERRS: collections.Counter = collections.Counter()


def _call(source: str, summary: str) -> dict | None:
    for _ in range(2):
        try:
            r = litellm.completion(
                model=MODEL,
                api_key=os.environ["ANTHROPIC_API_KEY"],
                extra_headers=HDR,
                max_tokens=3000,  # haiku-5-5 spends hidden reasoning tokens; 120 returns content=None
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"SOURCE:\n{source}\n\nSUMMARY:\n{summary}"},
                ],
            )
            m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S)
            if m:
                return json.loads(m.group(0))
            ERRS["no_json"] += 1
        except Exception as e:  # count kinds, never swallow silently
            ERRS[type(e).__name__] += 1
    return None


def judge_all(items: list[dict], seed: int = 3) -> list[dict]:
    jobs = [(i, s) for i, it in enumerate(items) for s, t in it["systems"].items() if t and t.strip()]
    random.Random(seed).shuffle(jobs)
    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(lambda j: _call(items[j[0]]["source"], items[j[0]]["systems"][j[1]]), jobs))
    for (i, s), r in zip(jobs, res, strict=False):
        items[i].setdefault("judge", {})[s] = r
    return items


def summarise(items: list[dict], seed: int = 0) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    per: dict[str, dict] = collections.defaultdict(lambda: {"sup": 0, "tot": 0, "ff": [], "flu": []})
    for it in items:
        for s, j in it.get("judge", {}).items():
            if not j:
                continue
            v = [c.get("verdict") for c in j.get("claims", [])]
            d = per[s]
            d["sup"] += v.count("supported")
            d["tot"] += len(v)
            d["ff"].append(1.0 if v and all(x == "supported" for x in v) else 0.0)
            d["flu"].append(float(j.get("fluency", 0)))
    out = {}
    for s, d in per.items():
        means = sorted(st.mean(rng.choices(d["ff"], k=len(d["ff"]))) for _ in range(2000))
        out[s] = {
            "n": len(d["ff"]),
            "claims_supported": 100 * d["sup"] / max(d["tot"], 1),
            "fully_faithful": 100 * st.mean(d["ff"]),
            "ff_ci_lo": 100 * means[50],
            "ff_ci_hi": 100 * means[1949],
            "fluency": st.mean(d["flu"]),
        }
    return out


def main() -> None:
    items = json.load(open(sys.argv[1]))
    items = judge_all(items)
    json.dump(items, open(sys.argv[2], "w"))
    print(f"judge errors: {dict(ERRS) or 'none'}")
    print(f"{'system':<14}{'n':>5} {'claims supported':>17} {'fully faithful [95% CI]':>26} {'fluency':>8}")
    for s, r in sorted(summarise(items).items()):
        print(
            f"{s:<14}{r['n']:>5} {r['claims_supported']:>16.0f}% "
            f"{r['fully_faithful']:>8.0f}% [{r['ff_ci_lo']:.0f}-{r['ff_ci_hi']:.0f}] {r['fluency']:>12.1f}"
        )
    anchors = summarise(items)
    bad = [
        n
        for n, lo, hi in (("lead1", 90, 101), ("mismatch", -1, 10))
        if n in anchors and not lo <= anchors[n]["claims_supported"] <= hi
    ]
    print("ANCHORS OK" if not bad and {"lead1", "mismatch"} <= set(anchors) else f"ANCHOR PROBLEM: {bad or 'missing'}")


if __name__ == "__main__":
    main()
