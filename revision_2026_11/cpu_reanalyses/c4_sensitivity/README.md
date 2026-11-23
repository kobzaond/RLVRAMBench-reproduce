# C4–C6: eligibility sensitivity, uncertainty, and GPU time

Run in order with `RLVRAMBENCH_TABLES_ROOT` and `RLVRAMBENCH_ANALYSIS_OUTPUT`
set (all four write into the same output directory):

1. `c4_eligibility.py` (C4): slot outcomes, labels, transfer scores and paired
   contrasts under the published rule and two alternative eligibility rules.
2. `c4_predictors.py` (C4): predictor comparisons after refitting on the
   configurations that stay resolved under each rule; leave-one-out sensitivity.
   Needs `c4_configuration_labels_by_rule.csv` from step 1.
3. `c5_uncertainty.py` (C5): case-level bootstrap (10,000 draws of whole
   model–workload cases, seed 20261110) and exact Clopper–Pearson bounds.
4. `c6_gpu_time.py` (C6): elapsed GPU time of donor evidence, screens, and
   evaluations in the budgeted-admission comparison.
