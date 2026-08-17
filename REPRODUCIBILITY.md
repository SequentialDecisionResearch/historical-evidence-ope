# Reproducibility Guide

This document describes how to reproduce the public v3.1 experiment from the source code in this repository.

## 1. What is archived here

The repository includes:

- the public-release Python script;
- compact summary tables from the original v3.1 run;
- publication-ready copies of the main figures;
- the public-facing article.

Raw QQQ market data are not archived. They are downloaded locally by the script.

## 2. Original run

The archived console report records:

- Python: `3.13.5`
- yfinance: `0.2.65`
- raw rows: `6,694`
- processed rows: `6,673`
- processed period: `2000-02-01` to `2026-08-13`
- master behavior seeds: `20260814`, `20260815`, `20260816`
- primary logging minimum action probability: `0.03`

The script writes a fresh `data/metadata/environment_and_config.json` file on every run so that the local Python, platform, configuration, and installed package versions are recorded.

The supplied historical console report did not preserve exact versions for every scientific Python dependency, so this repository does not invent version pins that were not recorded. `yfinance==0.2.65` is pinned because that version is explicitly present in the archived run report.

## 3. Installation

Create or activate a Python environment, then install:

```bash
pip install -r requirements.txt
```

The required packages are:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- yfinance
- requests

`requests` is used only by the optional Alpha Vantage cross-check helper.

## 4. Run the experiment

From the repository root:

```bash
python src/historical_evidence_ope.py
```

Spyder users can open `src/historical_evidence_ope.py` and use **Run File**.

The public-release copy differs from the original local script only in repository portability settings:

1. the output directory defaults to the repository root instead of a personal Windows path;
2. the data cutoff is fixed at `2026-08-14` in the download request, matching the original processed history through `2026-08-13`.

The research logic is otherwise unchanged.

## 5. Generated directory structure

A successful run creates:

```text
data/
├── raw/
├── processed/
└── metadata/

results/
├── figures/
├── tables/
└── logs/
```

Important generated metadata include:

- `environment_and_config.json`
- `download_metadata_yfinance.json`
- `data_quality.json`
- `log_sanity_checks.json`
- `chosen_parameters.json`
- `candidate_policy_distance.json`

Important generated data include:

- `qqq_market_features.csv`
- `qqq_simulated_bandit_logs.csv`

These generated market-data files should remain local and should not be committed to the public repository.

## 6. Main experimental outputs

The script runs the following sequence:

1. download/load QQQ data;
2. construct leakage-safe market features;
3. generate the Thompson-sampling behavior log;
4. tune memory, re-logging size, and ESS-fraction settings on validation anchors;
5. evaluate standard OPE estimators against the hidden oracle;
6. compare historical-memory rules;
7. compare recent old-policy evidence with simulated target-aware re-logging;
8. evaluate a candidate-baseline update rule using moving-block bootstrap, lower confidence bounds, and ESS;
9. sweep the logging exploration floor;
10. repeat the main experiments across independent behavior-policy seeds;
11. run a controlled synthetic lower-confidence-bound stress test.

The main summary CSV files are copied into `results/tables/`.

## 7. Expected sanity checks

The original archived run reported:

```text
rows: 6673
minimum_any_action_probability: 0.03
minimum_observed_propensity: 0.03
max_reward_oracle_mismatch: 0.0
position_min: 0.0
position_max: 1.0
passed: true
```

Small floating-point differences are possible, but the selected-action reward must match the corresponding hidden-oracle reward and behavior probabilities must sum to one.

## 8. Fast synthetic smoke test

The source file includes `synthetic_smoke_test()`.

For a quick offline logic check:

1. open `src/historical_evidence_ope.py`;
2. at the bottom, comment out:

```python
main()
```

3. uncomment:

```python
synthetic_smoke_test()
```

4. run the file.

The upload-ready public copy was syntax-checked and its synthetic smoke test completed successfully before packaging.

This smoke test is only a software check. It does not reproduce the paper results.

## 9. Why a rerun can differ

Even with fixed random seeds, a future rerun can differ if a third-party data provider revises historical values, adjustment conventions, or API behavior.

For that reason:

- the public code pins the publication cutoff date;
- the script records download metadata and file hashes locally;
- compact result tables and figures from the archived run are included in `results/`.

The archived tables should be treated as the record of the reported v3.1 experiment. A fresh rerun is a reproducibility check, not a guarantee that an external data service will return byte-identical history indefinitely.

## 10. Reproducing the figures

Run the full pipeline. The script writes PNG files to:

```text
results/figures/
```

The public repository also contains clean copies of the main figures used for communication. Their names are simplified for readers; the source script may generate internal names such as `F3b_...` or `F9a_...`.

## 11. Research limitations

The hidden oracle is one-step only and is conditional on the logged pre-action position. It is not a full alternative trajectory.

The behavior logs are simulated.

Validation uses the hidden oracle to compare some evidence rules.

The same underlying QQQ sequence is reused across several robustness runs.

These limitations are part of the experimental design and should be retained when the results are cited or extended.
