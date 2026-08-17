# Results

This directory contains a compact public record of the v3.1 empirical results.

The full script generates more diagnostic files than are necessary for the repository landing page. The initial public release keeps the summary tables and the figures that support the main paper narrative.

## Tables

- `exp01_ope_summary.csv` — standard OPE estimators versus the hidden one-step oracle.
- `exp02_memory_summary.csv` — historical-memory rules.
- `exp03_recent_evidence_summary.csv` — recent old-policy evidence versus target-aware re-logging.
- `exp04_safe_update_summary.csv` — candidate-baseline update diagnostics.
- `exp05_exploration_summary.csv` — exploration, ESS, OPE error, and behavior-reward summaries.
- `robustness_summary_compact.csv` — compact cross-seed robustness results.
- `robustness_chosen_parameters_by_seed.csv` — validation-selected settings by behavior seed.
- `supp_synthetic_lcb_stress_summary.csv` — controlled lower-confidence-bound stress test.

## Figures

- `ess_vs_ope_error.png`
- `memory_rule_mae.png`
- `recent_evidence_error.png`
- `recent_evidence_ess.png`
- `lcb_adoption_power.png`
- `exploration_ess.png`
- `exploration_ope_error.png`

The filenames are reader-facing names. The source script may generate internal publication-development names such as `F3b_...`, `F4_...`, or `F9a_...`.

## Interpretation

These files document one empirical laboratory. They should not be treated as evidence for universal parameter choices such as a fixed memory horizon, a fixed re-logging budget, or a universally optimal exploration rate.
