# Case-Fold Tuning Report

## Study Summary

| Item | Value |
| --- | --- |
| Status | completed |
| Study name | case_fold_701515_study |
| Optuna completed trial count | 120 |
| Objective-completed trial count | 120 |
| Trusted trial count | 57 |
| Best params artifact written | True |
| Best trial | 53 |
| Best value | 0.967 |
| Objective metric | pairwise_f_beta |
| Objective beta | 0.500 |
| Best keep threshold | 0.590 |
| Best neutral threshold | 0.820 |
| Recall guardrail | Pairwise recall >= 0.700 per fold |

## Best Params

- `colsample_bytree`: `0.810973891770281`
- `learning_rate`: `0.04759218120544132`
- `min_child_samples`: `13`
- `n_estimators`: `400`
- `num_leaves`: `52`
- `reg_lambda`: `2.840199837199106`
- `subsample`: `0.9814546928945571`

## Winning Trial Fold Metrics

| Fold | Held-out case | Pairwise F-beta | Pairwise beta | Keep threshold | Neutral threshold | Pairwise P | Pairwise R | Pairwise F1 | Pairwise F0.5 | B-cubed F0.5 | B-cubed P | B-cubed R | ARI | NMI | Matching P | Matching R | Matching F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_epstein_held_out | case_epstein_real_01 | 0.950 | 0.500 | 0.590 | 0.820 | 1.000 | 0.790 | 0.883 | 0.950 | 0.946 | 1.000 | 0.779 | 0.873 | 0.937 | 1.000 | 0.704 | 0.826 |
| fold_palme_held_out | case_palme_real_01 | 0.987 | 0.500 | 0.590 | 0.820 | 1.000 | 0.939 | 0.969 | 0.987 | 0.992 | 1.000 | 0.961 | 0.966 | 0.987 | 1.000 | 0.939 | 0.969 |
| fold_tall_pines_held_out | case_tall_pines_01 | 0.964 | 0.500 | 0.590 | 0.820 | 0.982 | 0.898 | 0.938 | 0.964 | 0.966 | 0.982 | 0.905 | 0.934 | 0.970 | 0.699 | 0.980 | 0.816 |

## Final Hold-Out Test

| Case | Pairwise F-beta | Pairwise beta | Pairwise P | Pairwise R | B-cubed R | Run ID |
| --- | --- | --- | --- | --- | --- | --- |
| styles_case_manual | 0.894 | 0.500 | 0.894 | 0.894 | 0.959 | 0249ae1534a2859bdd3c0bd78912fdda |

Geometric mean pairwise F-beta: 0.894
