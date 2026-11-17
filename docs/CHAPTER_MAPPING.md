# Which artefact serves which part of the chapter

Every quantitative statement in the chapter is a macro written to
`chapter/generated/numbers.tex`, and every results table a file in
`chapter/generated/`, by `scripts/make_chapter_assets.py` from the run logs;
nothing is typed by hand. The same values, and every qualitative claim the text
makes about them, are written to `results/chapter_numbers.json`, which
`scripts/verify_chapter.py` checks. (The manuscript itself is not distributed
here before publication.)

| Chapter element | Experiment manifest(s) | Script that runs it |
|---|---|---|
| Table 2 (conventional comparison), §5.2 | `headline`, `ablation_powered` | `run_experiments.py`, `run_robustness.py` |
| Table 3, Fig. 3(a) (Control 1), §5.4 (Control 2) | `matched_budget` (includes `tuned_baseline`) | `run_robustness.py` |
| Fig. 3(b), Table 6 (Control 3, drift structure) | `review_scope` (reuses `severity`, `stationary_control`, `tuned_stationary`) | `run_review_analyses.py` |
| Table 4 (Control 4) | `noise_probe` | `run_noise_probe.py` |
| Table 5 (partition sensitivity) | `review_partition` | `run_review_analyses.py` |
| §5.9 scalability | `scalability` | `run_experiments.py` |
| Fig. 2 (drift bands) | none: drawn from the simulator, seed 0 | `make_chapter_assets.py` |
| Fig. 1 (pipeline) | none: hand-written TikZ, part of the manuscript | — |

Section and table numbers refer to the revised chapter as compiled by
`scripts/build_chapter.py`.
