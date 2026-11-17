# Review analyses

Requested by the chapter reviewers. Numbers are as measured; see scripts/run_review_analyses.py for the design of each analysis.

## A. Partition design (n = 10 seeds, kappa = 1.5)

### ref rates (0.0001, 0.0005, 0.002)

- depth (hand-designed): first-40 = 46.48, last-20 = 55.86
- inverted: first-40 = 34.25 (delta -12.23, p_holm 0.0088); last-20 = 47.98 (delta -7.88, p_holm 0.0088)
- random: first-40 = 40.91 (delta -5.57, p_holm 0.0088); last-20 = 52.39 (delta -3.47, p_holm 0.0088)
- grad_snr: first-40 = 33.09 (delta -13.38, p_holm 0.0088); last-20 = 35.08 (delta -20.78, p_holm 0.0088)
- grad_snr agrees with the depth assignment on 39.4% of 360 tensor assignments (chance for an equal-size random partition: 33.3%).

### tuned rates (0.001, 0.003, 0.008)

- depth (hand-designed): first-40 = 50.51, last-20 = 58.50
- inverted: first-40 = 45.45 (delta -5.06, p_holm 0.0088); last-20 = 52.40 (delta -6.10, p_holm 0.0088)
- random: first-40 = 48.52 (delta -1.99, p_holm 0.0088); last-20 = 56.36 (delta -2.14, p_holm 0.0166)
- grad_snr: first-40 = 47.75 (delta -2.76, p_holm 0.0088); last-20 = 54.40 (delta -4.10, p_holm 0.0098)
- grad_snr agrees with the depth assignment on 38.9% of 360 tensor assignments (chance for an equal-size random partition: 33.3%).


## B. Drift severity and band separation (n = 10 seeds)

tau_med = 15 throughout. tau_slow = 50 is the default; 16 collapses the slow band onto the medium one; 150 widens the separation.

### conventional: nested_marl vs ippo

- kappa 0, tau_slow 50: nested 35.23 vs baseline 29.55, delta +5.67 (+19.2%), p 0.0029
- kappa 0.5, tau_slow 50: nested 34.55 vs baseline 28.14, delta +6.40 (+22.8%), p 0.0029
- kappa 1, tau_slow 50: nested 38.97 vs baseline 33.51, delta +5.46 (+16.3%), p 0.0029
- kappa 1.5, tau_slow 50: nested 46.48 vs baseline 41.26, delta +5.21 (+12.6%), p 0.0029
- kappa 2, tau_slow 50: nested 51.87 vs baseline 46.63, delta +5.24 (+11.2%), p 0.0029
- kappa 1.5, tau_slow 16: nested 29.26 vs baseline 23.77, delta +5.49 (+23.1%), p 0.0029
- kappa 1.5, tau_slow 150: nested 57.75 vs baseline 51.97, delta +5.79 (+11.1%), p 0.0029
- advantage at tau_slow 50 minus at 16: -0.28 (paired p 0.5941)
- advantage at tau_slow 50 minus at 150: -0.57 (paired p 0.3112)

### tuned: nested_cfg_high vs ippo_lr0p005

- kappa 0, tau_slow 50: nested 40.87 vs baseline 40.44, delta +0.43 (+1.1%), p 0.6605
- kappa 0.5, tau_slow 50: nested 41.31 vs baseline 41.10, delta +0.21 (+0.5%), p 0.7776
- kappa 1, tau_slow 50: nested 44.52 vs baseline 45.08, delta -0.56 (-1.2%), p 0.2351
- kappa 1.5, tau_slow 50: nested 50.51 vs baseline 49.60, delta +0.91 (+1.8%), p 0.0361
- kappa 2, tau_slow 50: nested 54.44 vs baseline 54.23, delta +0.21 (+0.4%), p 0.4478
- kappa 1.5, tau_slow 16: nested 36.26 vs baseline 36.14, delta +0.12 (+0.3%), p 0.8654
- kappa 1.5, tau_slow 150: nested 60.07 vs baseline 59.16, delta +0.91 (+1.5%), p 0.1180
- advantage at tau_slow 50 minus at 16: +0.79 (paired p 0.3522)
- advantage at tau_slow 50 minus at 150: +0.01 (paired p 0.9902)
