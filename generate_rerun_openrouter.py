"""Build configs_rerun_openrouter/ = every pass@3 config that has no result file,
re-pointed at the `openrouter` provider. Filenames, output paths and the short
model id are left untouched so the new result JSONs merge with the existing
results/ set (analyze_pass.py groups by configuration.llm.model).

Usage:  python generate_rerun_openrouter.py [--src configs_pass3_off] [--dst configs_rerun_openrouter]
"""
import argparse
import re
import shutil
from pathlib import Path

import yaml

RESULTS = Path("results")
_TS = re.compile(r"_\d{9,}$")

# Index result stems once (results/ holds >10k files; globbing per config is far too slow).
_RESULT_STEMS = {_TS.sub("", p.stem) for p in RESULTS.glob("*.json")}


def has_result(cfg: dict) -> bool:
    return Path(cfg["output"]["path"]).stem in _RESULT_STEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="configs_pass3_off")
    ap.add_argument("--dst", default="configs_rerun_openrouter")
    ap.add_argument("--provider_order", nargs="*", default=None,
                    help="Optional OpenRouter provider preference, e.g. Anthropic")
    ap.add_argument("--no_fallbacks", action="store_true",
                    help="With --provider_order: fail instead of falling back to another backend")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    n_total = n_missing = 0
    per_model = {}
    for y in sorted(src.glob("*.yaml")):
        n_total += 1
        cfg = yaml.safe_load(y.read_text(encoding="utf-8"))
        if has_result(cfg):
            continue
        n_missing += 1
        cfg["llm"]["name"] = "openrouter"
        if args.provider_order:
            cfg["llm"]["provider_order"] = args.provider_order
            if args.no_fallbacks:
                cfg["llm"]["allow_fallbacks"] = False
        (dst / y.name).write_text(yaml.safe_dump(cfg, default_flow_style=False), encoding="utf-8")
        m = cfg["llm"]["model"]
        per_model[m] = per_model.get(m, 0) + 1

    print(f"{n_total} configs in {src}, {n_missing} without results -> written to {dst}")
    for m, n in sorted(per_model.items()):
        print(f"  {m}: {n}")


if __name__ == "__main__":
    main()
