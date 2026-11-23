# C2/C3/C8: predictor diagnostics

Run in order (or `run_all.sh`), with `RLVRAMBENCH_TABLES_ROOT` and
`RLVRAMBENCH_ANALYSIS_OUTPUT` set:

1. `a_reproduce.py`: regenerates every frozen prediction with the repository's
   prediction code and recomputes the published scores; stops on any mismatch.
2. `b_hurdle.py` (C2): failure-aware hurdle predictors (startup model,
   runtime-OOM rule, conditional peak regression) scored by failure type.
3. `c_coverage.py` (C3): refit with the 48 independent-control configurations
   added; scored on the reservation-sensitivity panel.
4. `d_margins.py` (C8): baselines rescored at limits of 90–100% of capacity.

`common.py` holds the shared read-only loaders; it imports
`memory_tuner.estimation_baselines` and the two protocol adapters unchanged.
