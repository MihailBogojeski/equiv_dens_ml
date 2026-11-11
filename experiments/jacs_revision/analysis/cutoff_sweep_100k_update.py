#!/usr/bin/env python3
"""Readiness-gated refresh of the energy-head cutoff sweep (water PBE-D4, 4/5/6 A) in the letter, SI and main text
once the 100k-step continuations are finished AND scored.

Gate (all three cutoffs; nothing is edited unless every check passes):
  * results/revision/water_pbe_d4_avdz_energy_cutoff_<c>_100k/<run>/checkpoints/latest_checkpoint.pth stores step >= target
  * results/revision/energy_forces_v3/water_pbe_d4_avdz_cutoff_<c>_100k/summary.json exists, scores the four water
    splits with the expected structure counts, and names the same run directory
Progress files, log lines or the existence of a checkpoint file are NOT accepted as evidence of completion.

Without --apply the script only prints the status and, when ready, the 40k / 100k / production comparison and the
prose lines that mention the 40k budget (they must be rewritten by hand). With --apply it replaces the three sweep
rows of the letter table (tab:response-cutoff-energy) and of the SI table (tab:si_water_cutoff_ef), regenerates
cutoff_sweep_fullref.pdf from the 100k scores (copied to the letter and paper figure directories), and writes a
markdown report next to the scores. Prose is never edited automatically.

    .venv/bin/python experiments/jacs_revision/analysis/cutoff_sweep_100k_update.py [--apply] [--suffix _100k]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CUTS = (4, 5, 6)
SPLITS = [("id_test", 250), ("ood_size", 300), ("ood_order", 192), ("ood_density", 120)]
PROD = REPO / "results/revision/energy_forces_v2/water_pbe_d4_avdz_v3_slurm/summary.json"
DOCS = {"letter": REPO / "docs/response-to-jacs-review/jacs_response_plan_v3.tex",
        "si": REPO / "docs/equiv-ml-paper/SI/SI_v11.tex",
        "main": REPO / "docs/equiv-ml-paper/main_v11.tex"}
TABLE_LABELS = {"letter": "tab:response-cutoff-energy", "si": "tab:si_water_cutoff_ef"}
PROSE_PATTERNS = [r"4\\times10\^4", r"\b40k\b", r"matched energy-training budget", r"fewer energy-training steps",
                  r"production training budget", r"at \$10\^5\$ steps"]


def checkpoint_step(run_dir: Path) -> int | None:
    f = run_dir / "checkpoints" / "latest_checkpoint.pth"
    if not f.exists():
        return None
    import torch
    ck = torch.load(f, map_location="cpu", weights_only=False)
    return int(ck.get("step", -1))


def status(suffix: str, target: int, run_suffix: str | None = None) -> tuple[bool, list[str], dict]:
    ok, lines, scores = True, [], {}
    run_suffix = suffix if run_suffix is None else run_suffix
    for c in CUTS:
        runs = sorted((REPO / f"results/revision/water_pbe_d4_avdz_energy_cutoff_{c}{run_suffix}").glob("*/"))
        run = runs[-1] if runs else None
        step = checkpoint_step(run) if run else None
        summ_path = REPO / f"results/revision/energy_forces_v3/water_pbe_d4_avdz_cutoff_{c}{suffix}/summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.exists() else None
        problems = []
        if step is None or step < target:
            problems.append(f"checkpoint step {step} < {target}")
        if summ is None:
            problems.append("no summary.json")
        else:
            for sp, n in SPLITS:
                got = summ.get("by_split", {}).get(sp, {}).get("n")
                if got != n:
                    problems.append(f"{sp}: n={got}, expected {n}")
            if run and summ.get("run") and Path(summ["run"]).name != run.name:
                problems.append(f"summary.run {summ['run']} is not {run.name}")
        lines.append(f"cutoff {c} A: run={run.name if run else None} step={step} scores={'yes' if summ else 'no'}"
                     + (f"  PROBLEMS: {'; '.join(problems)}" if problems else "  ready"))
        ok &= not problems
        scores[c] = summ
    return ok, lines, scores


def fmt_e(x: float) -> str:
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def row(summ: dict) -> str:
    return " & ".join(f"{fmt_e(summ['by_split'][sp]['energy_mae_per_water_kcal_mol'])} / {summ['by_split'][sp]['force_mae_kcal_mol_A']:.2f}"
                      for sp, _ in SPLITS)


def replace_rows(tex: str, label: str, rows: dict[int, str]) -> tuple[str, int]:
    """Replace the lines starting with '4 &', '5 &', '6 &' inside the tabular that carries `label`."""
    k = tex.index("\\label{" + label + "}")
    # the tabular inside the table/rtable environment that carries the label (the label may precede or follow it)
    env_a = max(tex.rfind("\\begin{table}", 0, k), tex.rfind("\\begin{rtable}", 0, k))
    env_b = min(x for x in (tex.find("\\end{table}", k), tex.find("\\end{rtable}", k)) if x >= 0)
    a = tex.find("\\begin{tabular}", env_a, env_b); b = tex.find("\\end{tabular}", a, env_b)
    assert env_a >= 0 and a >= 0 and b > a, f"no tabular found for {label}"
    body, n = tex[a:b], 0
    out = []
    for ln in body.split("\n"):
        m = re.match(r"^(\s*)([456]) & .*\\\\\s*$", ln)
        if m and int(m.group(2)) in rows:
            ln = f"{m.group(1)}{m.group(2)} & {rows[int(m.group(2))]} \\\\"; n += 1
        out.append(ln)
    return tex[:a] + "\n".join(out) + tex[b:], n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_100k", help="score-set suffix ('' = the 40k heads, for a dry test)")
    ap.add_argument("--target-step", type=int, default=100000)
    ap.add_argument("--run-suffix", default=None, help="run-directory suffix when it differs from the score suffix (e.g. _100k runs scored at 91k)")
    ap.add_argument("--no-tables", action="store_true", help="with --apply: figure and report only, leave the documents alone")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--docs-root", default=None, help="edit copies under this root instead of the repo documents (test)")
    args = ap.parse_args()
    ok, lines, scores = status(args.suffix, args.target_step, args.run_suffix if args.run_suffix is not None else args.suffix)
    print("\n".join(lines))
    if not ok:
        print("\nNOT READY: no document was changed."); return 2
    prod = json.loads(PROD.read_text()) if PROD.exists() else None
    old = {c: (REPO / f"results/revision/energy_forces_v3/water_pbe_d4_avdz_cutoff_{c}/summary.json") for c in CUTS}
    old = {c: json.loads(p.read_text()) for c, p in old.items() if p.exists()}
    hdr = "| head | " + " | ".join(sp for sp, _ in SPLITS) + " |\n|---|" + "---|" * len(SPLITS)
    rep = [f"# Energy-head cutoff sweep: {args.suffix.strip('_') or '40k'} scores (E per water kcal/mol / F kcal/mol/A)", "", hdr]
    for c in CUTS:
        if c in old:
            rep.append(f"| {c} A, 40k | " + row(old[c]).replace(" & ", " | ") + " |")
        rep.append(f"| {c} A, {args.suffix.strip('_')} | " + row(scores[c]).replace(" & ", " | ") + " |")
    if prod:
        rep.append("| 6 A production (100k) | " + row(prod).replace(" & ", " | ") + " |")
    rep += ["", "Bias-removed energy MAE per cluster (kcal/mol):", hdr]
    for c in CUTS:
        rep.append(f"| {c} A, {args.suffix.strip('_')} | " + " | ".join(f"{scores[c]['by_split'][sp]['energy_mae_bias_removed_kcal_mol']:.3f}" for sp, _ in SPLITS) + " |")
    rep += ["", "Prose lines that mention the 40k budget (rewrite by hand after reading the numbers):"]
    root = Path(args.docs_root) if args.docs_root else None
    for name, path in DOCS.items():
        p = (root / path.relative_to(REPO)) if root else path
        for i, ln in enumerate(p.read_text().split("\n"), 1):
            if any(re.search(pat, ln) for pat in PROSE_PATTERNS):
                rep.append(f"- {name} {path.name}:{i}: {ln.strip()[:140]}")
    print("\n".join(rep))
    if not args.apply:
        print("\n(dry run: pass --apply to replace the table rows, regenerate the figure and write the report)"); return 0
    rows = {c: row(scores[c]) for c in CUTS}
    for name, label in ({} if args.no_tables else TABLE_LABELS).items():
        p = (root / DOCS[name].relative_to(REPO)) if root else DOCS[name]
        new, n = replace_rows(p.read_text(), label, rows)
        p.write_text(new); print(f"{p}: {n} rows replaced in {label}")
    out = REPO / f"results/revision/energy_forces_v3/cutoff_sweep{args.suffix}_update.md"
    out.write_text("\n".join(rep) + "\n"); print("wrote", out)
    if not root:
        env = {**os.environ, "CUTOFF_ENERGY_SUFFIX": args.suffix, "CUTOFF_STEPS": args.suffix.strip("_") or "40k"}
        subprocess.run([sys.executable, str(REPO / "experiments/jacs_revision/analysis/plot_response_figures_v2.py"), "fig_cutoff"],
                       check=True, env=env, cwd=REPO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
