# Control 4: precondition probe

Channel noise and demand jitter scaled together over [0.1, 0.3, 0.5] (default 0.10); n = 10 seeds, 80 episodes. Cells are first-40 / last-20 mean reward.

| Noise | IPPO ($\alpha$=0.005) | IPPO ($\alpha$=0.002) | Nested-MARL (ours) |
|---|---|---|---|
| 0.10 (default) | 49.60 / 57.88 | 49.23 / 57.63 | 46.48 / 55.86 |
| 0.30 | 33.48 / 40.39 | 31.88 / 39.63 | 29.56 / 37.32 |
| 0.50 | 25.00 / 31.11 | 24.45 / 29.98 | 21.78 / 28.94 |

Nested minus IPPO (alpha = 5e-3), first-40 reward:

- noise 0.10: delta -3.12 (-6.3%), p 0.0029
- noise 0.30: delta -3.92 (-11.7%), p 0.0029
- noise 0.50: delta -3.22 (-12.9%), p 0.0029

> **Result: the precondition is not restored.** The tuned single-rate baseline leads at every noise level.
