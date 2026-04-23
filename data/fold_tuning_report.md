# Case-Fold Tuning Report

## Study Summary

| Item | Value |
| --- | --- |
| Status | completed |
| Study name | case_fold_eval2_full |
| Optuna completed trial count | 120 |
| Objective-completed trial count | 120 |
| Trusted trial count | 3 |
| Best params artifact written | True |
| Best trial | 34 |
| Best value | 0.955 |
| Objective metric | pairwise_f_beta |
| Objective beta | 0.500 |
| Best keep threshold | 0.480 |
| Best neutral threshold | 0.700 |
| Recall guardrail | Pairwise recall >= 0.700 per fold |

## Best Params

- `colsample_bytree`: `0.94996143259553`
- `learning_rate`: `0.029766071081501526`
- `min_child_samples`: `94`
- `n_estimators`: `300`
- `num_leaves`: `23`
- `reg_lambda`: `11.439984343284452`
- `subsample`: `0.9799003358894236`

## Winning Trial Fold Metrics

| Fold | Held-out case | Pairwise F-beta | Pairwise beta | Keep threshold | Neutral threshold | Pairwise P | Pairwise R | Pairwise F1 | Pairwise F0.5 | B-cubed F0.5 | B-cubed P | B-cubed R | ARI | NMI | Matching P | Matching R | Matching F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_epstein_held_out | case_epstein_real_01 | 0.948 | 0.500 | 0.480 | 0.700 | 1.000 | 0.785 | 0.880 | 0.948 | 0.943 | 1.000 | 0.769 | 0.869 | 0.933 | 1.000 | 0.704 | 0.826 |
| fold_palme_held_out | case_palme_real_01 | 0.987 | 0.500 | 0.480 | 0.700 | 1.000 | 0.939 | 0.969 | 0.987 | 0.992 | 1.000 | 0.961 | 0.966 | 0.987 | 1.000 | 0.939 | 0.969 |
| fold_tall_pines_held_out | case_tall_pines_01 | 0.959 | 0.500 | 0.480 | 0.700 | 0.982 | 0.877 | 0.926 | 0.959 | 0.962 | 0.982 | 0.887 | 0.922 | 0.966 | 0.966 | 0.926 | 0.946 |
| fold_styles_held_out | styles_case_manual | 0.928 | 0.500 | 0.480 | 0.700 | 0.937 | 0.894 | 0.915 | 0.928 | 0.975 | 0.978 | 0.959 | 0.913 | 0.986 | 0.925 | 0.939 | 0.932 |
