#!/usr/bin/env python3
"""Evaluation controls for the multi-timescale claim.

Each experiment is designed so that a negative outcome would undermine the
claim that separating learning rates helps under multi-timescale drift.

**R1 -- Tuned-baseline control.** IPPO is swept across the rates the nested
method uses internally (1e-4, 5e-4, 1e-3, 2e-3, 5e-3) and the reference nested
configuration is compared with the *best* of them. If a single-rate IPPO matches
it, the advantage over the conventional baseline is a learning-rate effect.

**R2 -- Stationary control (kappa = 0).** If the benefit comes from tracking
drift, removing the drift should remove the advantage. An advantage that
persists at kappa = 0 comes from something else.

**R3 -- Powered ablations.** At n = 5 the permutation-test floor is 0.09, so the
ablations cannot reach significance at alpha = 0.05; n = 10 lowers it to 0.003.

**R4 -- Matched tuning budget.** R1 tunes the baseline only. R4 gives both
families five configurations each and compares the best of each.

**R5 -- Tuned severity control.** R2 with the R4 winners, across all
severities. The winners were selected at kappa = 1.5, so the other severities
are out of sample.

Usage
-----
    python scripts/run_robustness.py --workers 4
    python scripts/run_robustness.py --workers 4 --report-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nested6g.agents.methods import (METHODS, NESTED_SWEEP_METHODS,  # noqa: E402
                                     SWEEP_METHODS)
from nested6g.analysis import aggregate as agg  # noqa: E402
from nested6g.analysis.statistics import holm_bonferroni, summarise  # noqa: E402
from nested6g.experiments import SEEDS  # noqa: E402
from nested6g.runner import (RESULTS_ROOT, RunSpec, load_experiment,  # noqa: E402
                             run_many, write_manifest)

# --------------------------------------------------------------------------- #
#  R1: register IPPO at each rate the nested method uses internally
# --------------------------------------------------------------------------- #
# The swept IPPO variants are registered permanently in
# nested6g.agents.methods, so that spawned worker processes see them too.
# (Registering them at runtime here would not survive process spawn on
# macOS or Windows, where workers re-import the package from scratch.)


def register_sweep_methods() -> list[str]:
    """Names of the swept IPPO variants."""
    return list(SWEEP_METHODS)


# --------------------------------------------------------------------------- #
#  Grids
# --------------------------------------------------------------------------- #
def specs(episodes: int, n_seeds: int) -> dict[str, list[RunSpec]]:
    seeds = SEEDS[:n_seeds]
    sweep = register_sweep_methods()

    r1 = [
        RunSpec(experiment="tuned_baseline", method=m, seed=s,
                episodes=episodes, kappa=1.5)
        for m in [*sweep, "nested_marl"]
        for s in seeds
    ]
    r2 = [
        RunSpec(experiment="stationary_control", method=m, seed=s,
                episodes=episodes, kappa=k)
        for m in ("ippo", "nested_marl")
        for k in (0.0, 1.5)
        for s in seeds
    ]
    r3 = [
        RunSpec(experiment="ablation_powered", method=m, seed=s,
                episodes=episodes, kappa=1.5)
        for m in ("ippo", "nested_marl", "nested_same_rate",
                  "nested_two_timescale", "nested_no_ema")
        for s in seeds
    ]
    r4 = [
        RunSpec(experiment="matched_budget", method=m, seed=s,
                episodes=episodes, kappa=1.5)
        for m in [*sweep, *NESTED_SWEEP_METHODS]
        for s in seeds
    ]
    # R5 needs R4's winners, which are only known after R4 has run. The grid
    # is therefore built lazily by `tuned_stationary_specs`.
    return {"tuned_baseline": r1, "stationary_control": r2,
            "ablation_powered": r3, "matched_budget": r4}


TUNED_KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def tuned_stationary_specs(root: Path, episodes: int, n_seeds: int):
    """Grid for R5, using the best configuration of each family from R4."""
    s = agg.per_seed_summary(load_experiment("matched_budget", root))
    def best(names):
        m = {k: s[s["method"] == k]["reward_first40"].mean()
             for k in names if k in set(s["method"])}
        return max(m, key=m.get)
    winners = (best(SWEEP_METHODS), best(NESTED_SWEEP_METHODS))
    specs_ = [
        RunSpec(experiment="tuned_stationary", method=m, seed=sd,
                episodes=episodes, kappa=k)
        for m in winners for k in TUNED_KAPPAS for sd in SEEDS[:n_seeds]
    ]
    return winners, specs_


def analyse_r5(root: Path, md: list[str]) -> None:
    df = load_experiment("tuned_stationary", root)
    s = agg.per_seed_summary(df, group_keys=("method", "kappa", "seed"))
    n = int(s.groupby(["method", "kappa"])["seed"].nunique().max())
    methods = sorted(set(s["method"]))
    nest = [m for m in methods if m in NESTED_SWEEP_METHODS]
    sing = [m for m in methods if m in SWEEP_METHODS]
    if not nest or not sing:
        md.append("## R5. Tuned stationary control\n\n_Incomplete._\n")
        return
    nest, sing = nest[0], sing[0]

    rows, res = [], {}
    for k in sorted(s["kappa"].unique()):
        a = s[(s["method"] == nest) & (s["kappa"] == k)].set_index("seed")
        b = s[(s["method"] == sing) & (s["kappa"] == k)].set_index("seed")
        sh = a.index.intersection(b.index)
        r = summarise(a.loc[sh, "reward_first40"].to_numpy(),
                      b.loc[sh, "reward_first40"].to_numpy(), f"k{k}")
        res[k] = r
        rows.append({"kappa": k, "tuned single-rate": r.mean_b,
                     "tuned nested": r.mean_a, "Delta": r.delta,
                     "Delta %": r.delta_pct, "p": r.p_value})

    md.append(f"## R5. Tuned stationary control (n = {n} seeds)\n")
    md.append(f"Best of each family from R4 ({METHODS[nest].label} vs "
              f"{METHODS[sing].label}), swept over drift severity. This is the "
              "decisive mechanism test.\n")
    md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f") + "\n")

    k0, kmax = min(res), max(res)
    d0, dmax = res[k0].delta, res[kmax].delta
    md.append(f"\nAdvantage at kappa={k0:g}: {d0:+.2f} "
              f"(p = {fmt_p(res[k0].p_value, n)}); at kappa={kmax:g}: "
              f"{dmax:+.2f} (p = {fmt_p(res[kmax].p_value, n)}).\n")

    if res[k0].p_value >= 0.05 and dmax > 0 and res[kmax].p_value < 0.05:
        md.append("\n> **Result: drift-dependent advantage.** With both families "
                  "tuned, the advantage is significant under drift and absent "
                  "without it, consistent with timescale tracking.\n")
    elif d0 > 0 and res[k0].p_value < 0.05:
        md.append("\n> **Result: not drift-dependent.** A tuned nested scheme "
                  "retains a significant advantage with no drift, so the edge "
                  "is not explained by timescale tracking.\n")
    else:
        md.append("\n> **Result: no drift-dependent advantage detected.** Neither "
                  "end of the severity range shows a significant advantage, "
                  "and the advantage does not grow with severity.\n")
    md.append("")


# --------------------------------------------------------------------------- #
#  Analysis
# --------------------------------------------------------------------------- #
def fmt_p(p: float, n: int) -> str:
    floor = 3.0 / (2 ** n + 1)
    return f"{p:.4f}" + (" (floor)" if p <= floor * 1.01 else "")


def analyse_r1(root: Path, md: list[str]) -> None:
    register_sweep_methods()
    df = load_experiment("tuned_baseline", root)
    s = agg.per_seed_summary(df)
    n = int(s.groupby("method")["seed"].nunique().max())

    rows = []
    for m in sorted(set(s["method"])):
        v = s[s["method"] == m]
        rows.append({
            "Method": METHODS[m].label if m in METHODS else m,
            "key": m,
            "First-40": v["reward_first40"].mean(),
            "All-80": v["reward_all80"].mean(),
            "Last-20": v["reward_last20"].mean(),
            "Switching": v["switching_last30"].mean(),
        })
    table = pd.DataFrame(rows).sort_values("First-40", ascending=False)

    md.append(f"## R1. Tuned-baseline control (n = {n} seeds)\n")
    md.append("Is the gain merely a better learning rate? IPPO is swept across "
              "every rate the nested method uses internally.\n")
    md.append(table.drop(columns="key").to_markdown(index=False, floatfmt=".2f") + "\n")

    best = table[table["key"] != "nested_marl"].iloc[0]
    nested = table[table["key"] == "nested_marl"].iloc[0]
    a = s[s["method"] == "nested_marl"].set_index("seed")["reward_first40"]
    b = s[s["method"] == best["key"]].set_index("seed")["reward_first40"]
    shared = a.index.intersection(b.index)
    res = summarise(a.loc[shared].to_numpy(), b.loc[shared].to_numpy(), "best_ippo")

    md.append(f"\n**Best single-rate IPPO:** {best['Method']} "
              f"(first-40 = {best['First-40']:.2f}).\n")
    md.append(f"**Nested-MARL:** {nested['First-40']:.2f}, "
              f"delta = {res.delta:+.2f} ({res.delta_pct:+.1f}%), "
              f"p = {fmt_p(res.p_value, n)}.\n")
    if res.delta > 0 and res.p_value < 0.05:
        md.append("\n> **Result: Nested-MARL beats the best single-rate IPPO.** "
                  "The advantage is not explained by the choice of learning "
                  "rate.\n")
    elif res.delta > 0:
        md.append("\n> **Result: directional, not significant.** Nested-MARL "
                  "leads the best-tuned baseline without reaching "
                  "significance.\n")
    else:
        md.append("\n> **Result: a tuned single-rate IPPO matches or beats "
                  "Nested-MARL.** The advantage over the conventional baseline "
                  "is explained by the baseline's learning rate.\n")
    md.append("")


def analyse_r2(root: Path, md: list[str]) -> None:
    df = load_experiment("stationary_control", root)
    s = agg.per_seed_summary(df, group_keys=("method", "kappa", "seed"))
    n = int(s.groupby(["method", "kappa"])["seed"].nunique().max())

    rows, verdict = [], {}
    for k in sorted(s["kappa"].unique()):
        a = s[(s["method"] == "nested_marl") & (s["kappa"] == k)].set_index("seed")
        b = s[(s["method"] == "ippo") & (s["kappa"] == k)].set_index("seed")
        shared = a.index.intersection(b.index)
        av = a.loc[shared, "reward_first40"].to_numpy()
        bv = b.loc[shared, "reward_first40"].to_numpy()
        res = summarise(av, bv, f"k{k}")
        rows.append({"kappa": k, "IPPO": bv.mean(), "Nested-MARL": av.mean(),
                     "Delta": res.delta, "Delta %": res.delta_pct,
                     "p": res.p_value})
        verdict[k] = res

    md.append(f"## R2. Stationary control (n = {n} seeds)\n")
    md.append("If the mechanism tracks hierarchical drift, removing the drift "
              "should remove the advantage.\n")
    md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f") + "\n")

    v0, v15 = verdict.get(0.0), verdict.get(1.5)
    if v0 is not None and v15 is not None:
        shrink = 100.0 * (1.0 - (v0.delta / v15.delta)) if v15.delta else float("nan")
        md.append(f"\nAdvantage at kappa=0: {v0.delta:+.2f} "
                  f"(p = {fmt_p(v0.p_value, n)}); at kappa=1.5: "
                  f"{v15.delta:+.2f}. The advantage shrinks by "
                  f"{shrink:.0f}% when drift is removed.\n")
        if v0.p_value >= 0.05 and v15.p_value < 0.05:
            md.append("\n> **Result: drift-dependent advantage.** The advantage is "
                      "significant under drift and absent without it.\n")
        elif v0.delta > 0 and v0.p_value < 0.05:
            md.append("\n> **Result: not drift-dependent.** A significant advantage "
                      "remains with no drift, so the gain is not explained by "
                      "timescale tracking.\n")
        else:
            md.append("\n> **Result: inconclusive.** Neither cell is clearly "
                      "significant.\n")
    md.append("")


def analyse_r3(root: Path, md: list[str]) -> None:
    df = load_experiment("ablation_powered", root)
    s = agg.per_seed_summary(df)
    n = int(s.groupby("method")["seed"].nunique().max())

    order = ["ippo", "nested_same_rate", "nested_two_timescale",
             "nested_no_ema", "nested_marl"]
    order = [m for m in order if m in set(s["method"])]
    rows = [{
        "Variant": METHODS[m].label,
        "First-40": s[s["method"] == m]["reward_first40"].mean(),
        "Last-30": s[s["method"] == m]["reward_last30"].mean(),
        "Switching": s[s["method"] == m]["switching_last30"].mean(),
    } for m in order]

    md.append(f"## R3. Powered ablation (n = {n} seeds)\n")
    md.append("At n=5 the permutation floor is 0.09, so ablations could not "
              "reach significance. At n=10 the floor is 0.003.\n")
    md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f") + "\n")

    comps = {}
    base = s[s["method"] == "nested_marl"].set_index("seed")["reward_first40"]
    for m in order:
        if m == "nested_marl":
            continue
        other = s[s["method"] == m].set_index("seed")["reward_first40"]
        shared = base.index.intersection(other.index)
        comps[m] = summarise(base.loc[shared].to_numpy(),
                             other.loc[shared].to_numpy(), m)
    adj = holm_bonferroni({k: v.p_value for k, v in comps.items()})

    md.append("\nNested-MARL vs each ablation (first-40 reward, Holm-corrected):\n")
    for m, r in comps.items():
        flag = "significant" if adj[m] < 0.05 else "NOT significant"
        md.append(f"- vs **{METHODS[m].label}**: delta = {r.delta:+.2f}, "
                  f"p_holm = {fmt_p(adj[m], n)} -> {flag}")

    sr = s[s["method"] == "nested_same_rate"]["reward_last30"].mean()
    ip = s[s["method"] == "ippo"]["reward_last30"].mean()
    md.append(f"\nSame-rate control: {sr:.2f} vs IPPO {ip:.2f} -> "
              f"{'BELOW' if sr < ip else 'above'} baseline.\n")
    if adj.get("nested_same_rate", 1.0) < 0.05:
        md.append("\n> **Result: within this configuration, removing timescale "
                  "separation significantly lowers reward.** The configuration "
                  "ablated here is itself beaten by a tuned single-rate "
                  "baseline (R1), so this does not establish a benefit over "
                  "tuned single-rate learning.\n")
    md.append("")


def analyse_r4(root: Path, md: list[str]) -> None:
    """Best-of-family vs best-of-family, with equal search budget."""
    df = load_experiment("matched_budget", root)
    s = agg.per_seed_summary(df)
    n = int(s.groupby("method")["seed"].nunique().max())

    def best_of(names, col="reward_first40"):
        means = {m: s[s["method"] == m][col].mean()
                 for m in names if m in set(s["method"])}
        return max(means, key=means.get), means

    bi, ippo_means = best_of(SWEEP_METHODS)
    bn, nest_means = best_of(NESTED_SWEEP_METHODS)

    rows = []
    for m in list(SWEEP_METHODS) + list(NESTED_SWEEP_METHODS):
        if m not in set(s["method"]):
            continue
        v = s[s["method"] == m]
        rows.append({
            "Family": "single-rate" if m in SWEEP_METHODS else "nested",
            "Configuration": METHODS[m].label,
            "First-40": v["reward_first40"].mean(),
            "Last-20": v["reward_last20"].mean(),
            "Switching": v["switching_last30"].mean(),
        })
    table = pd.DataFrame(rows).sort_values("First-40", ascending=False)

    md.append(f"## R4. Matched tuning budget (n = {n} seeds)\n")
    md.append("Both families receive five configurations. This is the fair "
              "comparison; R1 tuned only the baseline.\n")
    md.append(table.to_markdown(index=False, floatfmt=".2f") + "\n")

    a = s[s["method"] == bn].set_index("seed")["reward_first40"]
    b = s[s["method"] == bi].set_index("seed")["reward_first40"]
    shared = a.index.intersection(b.index)
    res = summarise(a.loc[shared].to_numpy(), b.loc[shared].to_numpy(), "best")

    md.append(f"\n**Best nested:** {METHODS[bn].label} "
              f"({nest_means[bn]:.2f}).  **Best single-rate:** "
              f"{METHODS[bi].label} ({ippo_means[bi]:.2f}).\n")
    md.append(f"**Delta = {res.delta:+.2f} ({res.delta_pct:+.1f}%), "
              f"p = {fmt_p(res.p_value, n)}.**\n")

    if res.delta > 0 and res.p_value < 0.05:
        md.append("\n> **Result: the best nested configuration leads under an "
                  "equal tuning budget at the selection severity.** Both "
                  "winners were selected at this severity, so the p-value is "
                  "optimistic; R5 evaluates them at the other severities.\n")
    elif res.delta > 0:
        md.append("\n> **Result: directional only.** The nested scheme leads "
                  "under equal tuning but not significantly.\n")
    else:
        md.append("\n> **Result: no advantage under equal tuning.** A tuned "
                  "single-rate baseline matches or beats the best nested "
                  "configuration.\n")

    # Secondary endpoint: does the stability advantage survive tuning?
    sa = s[s["method"] == bn].set_index("seed")["switching_last30"]
    sb = s[s["method"] == bi].set_index("seed")["switching_last30"]
    sres = summarise(sa.loc[shared].to_numpy(), sb.loc[shared].to_numpy(), "sw")
    md.append(f"\nPolicy churn (lower is better): nested {sa.mean():.3f} vs "
              f"tuned single-rate {sb.mean():.3f}, delta = {sres.delta:+.3f}, "
              f"p = {fmt_p(sres.p_value, n)}.\n")
    md.append("")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--results", type=Path, default=RESULTS_ROOT)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)

    grids = specs(args.episodes, args.seeds)
    if not args.report_only:
        total = sum(len(v) for v in grids.values())
        print("=" * 68)
        print(f"  Robustness suite: {total} runs "
              f"({args.seeds} seeds, {args.episodes} episodes)")
        print("  R1 tuned-baseline | R2 stationary control | R3 powered ablation")
        print("  Many runs are cache hits from the main suite.")
        print("=" * 68, flush=True)
        for name, g in grids.items():
            run_many(g, root=args.results, workers=args.workers, desc=name)
            write_manifest(name, g, root=args.results)
        # R5 depends on R4's winners, so its grid is built after R4 completes.
        winners, r5 = tuned_stationary_specs(args.results, args.episodes,
                                             args.seeds)
        print(f"[tuned_stationary] winners: {winners[1]} vs {winners[0]}")
        run_many(r5, root=args.results, workers=args.workers,
                 desc="tuned_stationary")
        write_manifest("tuned_stationary", r5, root=args.results)

    md = [
        "# Robustness report\n",
        "Three experiments that could falsify the central claim. Each states a "
        "verdict, including the verdicts that would require withdrawing a "
        "claim.\n",
    ]
    for fn in (analyse_r1, analyse_r4, analyse_r5, analyse_r2, analyse_r3):
        try:
            fn(args.results, md)
        except FileNotFoundError as exc:
            md.append(f"_Not available: {exc}_\n")

    out = args.results / "ROBUSTNESS.md"
    out.write_text("\n".join(md))
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
