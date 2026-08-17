# Data

Raw market data are not distributed in this repository.

The experiment downloads QQQ price history locally with `yfinance`, then constructs the features and one-step rewards needed by the contextual-bandit study.

## Publication data range

The archived v3.1 run used:

- symbol: `QQQ`
- requested start date: `2000-01-01`
- publication cutoff in the public script: `2026-08-14` (exclusive download end)
- final processed date: `2026-08-13`
- processed observations: `6,673`

## Generated local files

A full run creates local files under:

```text
data/raw/
data/processed/
data/metadata/
```

Examples include:

```text
data/raw/qqq_yfinance.csv
data/processed/qqq_market_features.csv
data/processed/qqq_simulated_bandit_logs.csv
data/metadata/environment_and_config.json
data/metadata/download_metadata_yfinance.json
data/metadata/data_quality.json
data/metadata/log_sanity_checks.json
```

These files are intentionally excluded from the public repository.

## Feature timing

The model context uses information available at the decision boundary:

```text
ret_1
ret_5
ret_20
vol_20
ma_spread
drawdown
position_before
```

`next_return` is used only to construct the one-step reward after the action. It is not part of the model feature vector.

## Third-party data

Market data remain subject to the original provider's terms. This repository publishes code, derived summary results, and figures rather than redistributing the raw third-party dataset.
