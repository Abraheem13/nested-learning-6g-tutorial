# Robustness report

Three experiments that could falsify the central claim. Each states a verdict, including the verdicts that would require withdrawing a claim.

## R1. Tuned-baseline control (n = 10 seeds)

Is the gain merely a better learning rate? IPPO is swept across every rate the nested method uses internally.

| Method                 |   First-40 |   All-80 |   Last-20 |   Switching |
|:-----------------------|-----------:|---------:|----------:|------------:|
| IPPO ($\alpha$=0.005)  |      49.60 |    53.47 |     57.88 |        0.18 |
| IPPO ($\alpha$=0.002)  |      49.23 |    53.27 |     57.63 |        0.16 |
| Nested-MARL (ours)     |      46.48 |    50.82 |     55.86 |        0.13 |
| IPPO ($\alpha$=0.001)  |      45.73 |    50.29 |     55.56 |        0.13 |
| IPPO ($\alpha$=0.0005) |      41.26 |    47.84 |     55.19 |        0.16 |
| IPPO ($\alpha$=0.0001) |      32.73 |    34.65 |     37.94 |        0.60 |


**Best single-rate IPPO:** IPPO ($\alpha$=0.005) (first-40 = 49.60).

**Nested-MARL:** 46.48, delta = -3.12 (-6.3%), p = 0.0029 (floor).


> **Result: a tuned single-rate IPPO matches or beats Nested-MARL.** The advantage over the conventional baseline is explained by the baseline's learning rate.


## R4. Matched tuning budget (n = 10 seeds)

Both families receive five configurations. This is the fair comparison; R1 tuned only the baseline.

| Family      | Configuration                  |   First-40 |   Last-20 |   Switching |
|:------------|:-------------------------------|-----------:|----------:|------------:|
| nested      | Nested (0.001, 0.003, 0.008)   |      50.51 |     58.50 |        0.18 |
| nested      | Nested (0.0002, 0.002, 0.01)   |      50.50 |     57.93 |        0.16 |
| nested      | Nested (0.002, 0.004, 0.008)   |      50.48 |     58.32 |        0.17 |
| single-rate | IPPO ($\alpha$=0.005)          |      49.60 |     57.88 |        0.18 |
| nested      | Nested (0.0005, 0.002, 0.005)  |      49.55 |     57.44 |        0.15 |
| single-rate | IPPO ($\alpha$=0.002)          |      49.23 |     57.63 |        0.16 |
| nested      | Nested (0.0001, 0.0005, 0.002) |      46.48 |     55.86 |        0.13 |
| single-rate | IPPO ($\alpha$=0.001)          |      45.73 |     55.56 |        0.13 |
| single-rate | IPPO ($\alpha$=0.0005)         |      41.26 |     55.19 |        0.16 |
| single-rate | IPPO ($\alpha$=0.0001)         |      32.73 |     37.94 |        0.60 |


**Best nested:** Nested (0.001, 0.003, 0.008) (50.51).  **Best single-rate:** IPPO ($\alpha$=0.005) (49.60).

**Delta = +0.91 (+1.8%), p = 0.0361.**


> **Result: the best nested configuration leads under an equal tuning budget at the selection severity.** Both winners were selected at this severity, so the p-value is optimistic; R5 evaluates them at the other severities.


Policy churn (lower is better): nested 0.179 vs tuned single-rate 0.182, delta = -0.003, p = 0.8302.


## R5. Tuned stationary control (n = 10 seeds)

Best of each family from R4 (Nested (0.001, 0.003, 0.008) vs IPPO ($\alpha$=0.005)), swept over drift severity. This is the decisive mechanism test.

|   kappa |   tuned single-rate |   tuned nested |   Delta |   Delta % |     p |
|--------:|--------------------:|---------------:|--------:|----------:|------:|
|   0.000 |              40.438 |         40.867 |   0.428 |     1.059 | 0.660 |
|   0.500 |              41.103 |         41.309 |   0.207 |     0.503 | 0.778 |
|   1.000 |              45.078 |         44.519 |  -0.559 |    -1.241 | 0.235 |
|   1.500 |              49.599 |         50.512 |   0.913 |     1.840 | 0.036 |
|   2.000 |              54.228 |         54.440 |   0.212 |     0.391 | 0.448 |


Advantage at kappa=0: +0.43 (p = 0.6605); at kappa=2: +0.21 (p = 0.4478).


> **Result: no drift-dependent advantage detected.** Neither end of the severity range shows a significant advantage, and the advantage does not grow with severity.


## R2. Stationary control (n = 10 seeds)

If the mechanism tracks hierarchical drift, removing the drift should remove the advantage.

|   kappa |   IPPO |   Nested-MARL |   Delta |   Delta % |     p |
|--------:|-------:|--------------:|--------:|----------:|------:|
|   0.000 | 29.554 |        35.225 |   5.672 |    19.191 | 0.003 |
|   1.500 | 41.265 |        46.476 |   5.212 |    12.630 | 0.003 |


Advantage at kappa=0: +5.67 (p = 0.0029 (floor)); at kappa=1.5: +5.21. The advantage shrinks by -9% when drift is removed.


> **Result: not drift-dependent.** A significant advantage remains with no drift, so the gain is not explained by timescale tracking.


## R3. Powered ablation (n = 10 seeds)

At n=5 the permutation floor is 0.09, so ablations could not reach significance. At n=10 the floor is 0.003.

| Variant                       |   First-40 |   Last-30 |   Switching |
|:------------------------------|-----------:|----------:|------------:|
| IPPO (baseline)               |     41.265 |    54.879 |       0.163 |
| Same rate (no timescale sep.) |     40.384 |    53.382 |       0.232 |
| Nested-MARL, 2 timescales     |     43.851 |    54.436 |       0.212 |
| Nested-MARL, no EMA           |     46.471 |    55.292 |       0.150 |
| Nested-MARL (ours)            |     46.476 |    55.495 |       0.134 |


Nested-MARL vs each ablation (first-40 reward, Holm-corrected):

- vs **IPPO (baseline)**: delta = +5.21, p_holm = 0.0117 -> significant
- vs **Same rate (no timescale sep.)**: delta = +6.09, p_holm = 0.0117 -> significant
- vs **Nested-MARL, 2 timescales**: delta = +2.63, p_holm = 0.0117 -> significant
- vs **Nested-MARL, no EMA**: delta = +0.01, p_holm = 0.9415 -> NOT significant

Same-rate control: 53.38 vs IPPO 54.88 -> BELOW baseline.


> **Result: within this configuration, removing timescale separation significantly lowers reward.** The configuration ablated here is itself beaten by a tuned single-rate baseline (R1), so this does not establish a benefit over tuned single-rate learning.

