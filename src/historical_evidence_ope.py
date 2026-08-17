"""
Historical Evidence + OPE research pipeline (public research release, v3.1)
============================================================================

PURPOSE
-------
This is a reproducible ONE-STEP contextual-bandit / off-policy-evaluation (OPE)
research program. QQQ is only an accessible experimental environment. The code
is NOT a brokerage system and does NOT place real trades.

The program is designed to support the five main empirical questions in the
working paper and public-facing article, plus publication-oriented robustness diagnostics:

1. Can standard OPE (DM / IPS / SNIPS / DR) recover a hidden one-step oracle?
2. Under nonstationarity, how should historical evidence be remembered?
3. Is recent history the same as new-policy rollout evidence? (No: the former
   mainly improves recency; the latter can also improve action coverage.)
4. Can Delta-V uncertainty + ESS gating reduce unsafe policy updates?
5. Does more exploration improve future evaluability, and what behavior-
   performance trade-off does it create?

ROBUSTNESS ADDED IN v3.1
------------------------
* Repeat the main Experiments 1-4 under multiple independently simulated
  Thompson-sampling behavior logs (master seeds).
* Increase exploration repeats so Experiment 5 is not driven by three lucky runs.
* Add an ESS-fraction gate, selected ONLY on validation anchors and then frozen.
* Add direct ESS-vs-OPE-error and recent-evidence-vs-ESS figures.
* Add a fast supplementary synthetic stress test showing when an LCB gate abstains
  or adopts as the true signal-to-noise ratio changes.

IMPORTANT SCOPE
---------------
* The hidden oracle is a ONE-STEP counterfactual oracle conditional on the
  logged context and the logged position-before-action. It is not a full
  multi-step counterfactual portfolio path.
* The discrete-time convention is: context x_t is observed at the t boundary,
  action a_t sets exposure for the t -> t+1 period, and reward is observed at
  t+1. This is a methodological RL laboratory, not an execution-price model.
* Before publication, independently verify data, code, figures and numerical
  conclusions. Do not claim that a method is better until repeated empirical
  checks support that statement.

DEFAULT WINDOWS OUTPUT DIRECTORY
--------------------------------
C:\\study_notes\\traval_rec\\QQQ_Contextual_Bandit_OPE

SPYDER
------
The file is a normal Python script and can be run from Spyder with Run File.
Figures are both saved to disk and, by default, shown with plt.show(block=False)
so they can appear in Spyder's Plots pane / graphics backend.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. Configuration
# =============================================================================

ACTIONS = np.array([-1, 0, 1], dtype=int)
ACTION_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_ACTION = {0: -1, 1: 0, 2: 1}

MARKET_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_20",
    "ma_spread",
    "drawdown",
]

MODEL_FEATURES = [
    "intercept",
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_20",
    "ma_spread",
    "drawdown",
    "position_before",
]

REGIME_FEATURES = ["vol_20", "ret_20", "drawdown"]


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    symbol: str = "QQQ"
    start_date: str = "2000-01-01"
    end_date: Optional[str] = "2026-08-14"  # yfinance end is exclusive; original run ends 2026-08-13
    seed: int = 20260814

    # Public-repository default: write generated files under the repository root.
    project_dir: str = str(Path(__file__).resolve().parents[1])

    # yfinance is the convenient prototype source. We require adjusted close
    # rather than silently switching to raw Close, because that would change
    # the reward definition.
    force_download: bool = False
    require_adjusted_close: bool = True

    # ------------------------------------------------------------------
    # Position / one-step reward
    # ------------------------------------------------------------------
    initial_position: float = 0.50
    position_step: float = 0.10
    min_position: float = 0.00
    max_position: float = 1.00
    transaction_cost: float = 0.0005
    risk_lambda: float = 0.00

    # ------------------------------------------------------------------
    # Behavior policy: Linear Thompson Sampling
    # ------------------------------------------------------------------
    prior_precision: float = 1.0
    assumed_reward_noise_std: float = 0.01
    ts_mc_draws: int = 400
    minimum_action_probability: float = 0.03

    # ------------------------------------------------------------------
    # Reward models and candidate target policies
    # ------------------------------------------------------------------
    ridge_alpha: float = 2.0
    long_policy_lookback: int = 756
    recent_policy_lookback: int = 126

    # IMPORTANT: target-policy softmax now uses a TRAINING-DATA q-scale so
    # temperature is dimensionless. This avoids the old failure mode where raw
    # daily rewards (~0.001) divided by a temperature like 0.7 made every policy
    # almost uniform.
    base_temperature: float = 1.50
    new_temperature: float = 0.70
    q_scale_floor: float = 0.00025
    policy_min_prob: float = 0.02
    conservative_eta: float = 0.20

    # Warn if base/new target policies are nearly identical; OPE then becomes a
    # trivial near-on-policy comparison and cannot support a strong article.
    minimum_mean_policy_l1_distance: float = 0.03

    # ------------------------------------------------------------------
    # Nonstationarity / historical memory
    # Values are TRADING OBSERVATIONS, not calendar days.
    # ------------------------------------------------------------------
    half_life_trading_days: int = 63
    regime_bandwidth: float = 1.0
    rolling_short: int = 60
    rolling_medium: int = 126
    rolling_long: int = 252

    # ------------------------------------------------------------------
    # Walk-forward anchors
    # ------------------------------------------------------------------
    min_history: int = 1260
    anchor_step: int = 63
    oracle_horizon: int = 80
    static_eval_window: int = 252

    # ------------------------------------------------------------------
    # Recent evidence / rollout
    # ------------------------------------------------------------------
    rollout_sizes: Tuple[int, ...] = (0, 5, 10, 20, 40, 80)
    rollout_new_policy_mix: float = 0.20

    # ------------------------------------------------------------------
    # Inference / conservative deployment gate
    # ------------------------------------------------------------------
    bootstrap_reps: int = 500
    bootstrap_block_length: int = 10
    confidence_level: float = 0.95  # one-sided LCB => z_0.95 ~ 1.645
    minimum_ess: float = 30.0

    # A normalized ESS requirement complements the absolute ESS requirement.
    # Unlike v3, v3.1 does NOT hard-code a test-driven fraction. It searches this
    # grid on VALIDATION anchors only, freezes the selected value, and then uses
    # that frozen value in Experiment 4 on untouched test anchors.
    minimum_ess_fraction: float = 0.0
    ess_fraction_gate_grid: Tuple[float, ...] = (0.00, 0.01, 0.02, 0.03, 0.05, 0.10)
    minimum_validation_adoption_rate: float = 0.10
    validation_bootstrap_reps: int = 80

    # ------------------------------------------------------------------
    # Validation grids: tune ONLY on validation anchors, then freeze.
    # ------------------------------------------------------------------
    half_life_grid: Tuple[int, ...] = (21, 63, 126, 252)
    regime_bandwidth_grid: Tuple[float, ...] = (0.5, 1.0, 2.0)
    validation_fraction: float = 0.60

    # ------------------------------------------------------------------
    # Exploration / coverage experiment
    # Repeat each epsilon under enough stochastic logging seeds for a Medium
    # robustness claim. 20 is the default compromise between stability and time.
    # ------------------------------------------------------------------
    exploration_epsilons: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
    exploration_ts_mc_draws: int = 150
    exploration_repeats: int = 20

    # ------------------------------------------------------------------
    # Master-seed robustness for Experiments 1-4. The first seed is the main
    # article run; additional seeds independently regenerate the historical
    # Thompson-sampling behavior log and repeat the chronological study.
    # ------------------------------------------------------------------
    run_master_seed_robustness: bool = True
    master_behavior_seeds: Tuple[int, ...] = (20260814, 20260815, 20260816)

    # Fast supplementary stress test for the LCB decision rule. This is NOT a
    # QQQ claim; it is a controlled diagnostic showing abstention/power as the
    # true Delta-V signal changes relative to correlated noise.
    run_synthetic_safety_stress_test: bool = True
    synthetic_safety_repeats: int = 250
    synthetic_safety_n: int = 252
    synthetic_safety_noise_std: float = 0.0010
    synthetic_safety_ar1: float = 0.35
    synthetic_safety_deltas: Tuple[float, ...] = (
        -0.00040, -0.00020, -0.00010, 0.0, 0.00010, 0.00020, 0.00040
    )

    # ------------------------------------------------------------------
    # User-facing output behavior
    # ------------------------------------------------------------------
    show_figures_in_spyder: bool = True
    save_figures: bool = True


# =============================================================================
# 2. Small utilities and reporting
# =============================================================================


def safe_package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def deterministic_rng(base_seed: int, anchor: int, salt: int = 0) -> np.random.Generator:
    """A reproducible local RNG so grid configurations use common random numbers."""
    seed = int((base_seed + 1_000_003 * int(anchor) + 97_003 * int(salt)) % (2**32 - 1))
    return np.random.default_rng(seed)


def ensure_project_dirs(cfg: Config) -> Dict[str, Path]:
    root = Path(cfg.project_dir)
    paths = {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "metadata": root / "data" / "metadata",
        "results": root / "results",
        "figures": root / "results" / "figures",
        "tables": root / "results" / "tables",
        "logs": root / "results" / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_environment_metadata(cfg: Config, paths: Dict[str, Path]) -> None:
    payload = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": asdict(cfg),
        "packages": {
            "numpy": safe_package_version("numpy"),
            "pandas": safe_package_version("pandas"),
            "scipy": safe_package_version("scipy"),
            "scikit-learn": safe_package_version("scikit-learn"),
            "matplotlib": safe_package_version("matplotlib"),
            "yfinance": safe_package_version("yfinance"),
        },
    }
    (paths["metadata"] / "environment_and_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def action_index(action: int) -> int:
    return ACTION_TO_INDEX[int(action)]


def clip_position(position: float, cfg: Config) -> float:
    return float(np.clip(position, cfg.min_position, cfg.max_position))


def stable_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-10)
    z = np.asarray(scores, dtype=float) / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def apply_probability_floor(probs: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply p <- (1-K*eps)*p + eps; preserves sum-to-one."""
    p = np.asarray(probs, dtype=float)
    k = p.shape[-1]
    if epsilon < 0 or epsilon >= 1.0 / k:
        raise ValueError(f"epsilon must satisfy 0 <= epsilon < {1.0 / k:.4f}")
    out = (1.0 - k * epsilon) * p + epsilon
    return out / np.sum(out, axis=-1, keepdims=True)


def model_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[MODEL_FEATURES].to_numpy(dtype=float)


def oracle_reward_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[["reward_oracle_m1", "reward_oracle_0", "reward_oracle_p1"]].to_numpy(
        dtype=float
    )


def observed_action_indices(df: pd.DataFrame) -> np.ndarray:
    return df["action"].map(ACTION_TO_INDEX).to_numpy(dtype=int)


def wealth_and_drawdown(rewards: Sequence[float], initial_wealth: float = 1.0) -> pd.DataFrame:
    r = np.asarray(rewards, dtype=float)
    wealth = initial_wealth * np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return pd.DataFrame({"reward": r, "wealth": wealth, "drawdown": drawdown})


class Reporter:
    """Print to Spyder console AND save the same important results to a text file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: List[str] = []

    def write(self, text: object = "") -> None:
        s = str(text)
        print(s)
        self.lines.append(s)

    def section(self, title: str) -> None:
        self.write("\n" + "=" * 88)
        self.write(title)
        self.write("=" * 88)

    def dataframe(self, df: pd.DataFrame, title: Optional[str] = None) -> None:
        if title:
            self.write(title)
        if df is None or df.empty:
            self.write("[empty table]")
        else:
            self.write(df.to_string(index=False))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines), encoding="utf-8")


def finalize_figure(path: Path, cfg: Config) -> None:
    plt.tight_layout()
    if cfg.save_figures:
        plt.savefig(path, dpi=180, bbox_inches="tight")
    if cfg.show_figures_in_spyder:
        # Works with Spyder's inline/automatic plotting backend. Non-blocking so
        # the script can continue through all experiments.
        plt.show(block=False)
        plt.pause(0.001)
    plt.close()


# =============================================================================
# 3. Data download, caching, versioning, and data-quality checks
# =============================================================================


def import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for the first download. In Spyder/Anaconda, run:\n"
            "    pip install yfinance\n"
            "or install it in the same environment used by Spyder."
        ) from exc
    return yf


def flatten_yfinance_columns(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw

    # Current yfinance commonly returns a MultiIndex even for one ticker. Try
    # selecting the ticker level; otherwise keep the price-field level.
    levels_last = raw.columns.get_level_values(-1)
    if symbol in levels_last:
        return raw.xs(symbol, axis=1, level=-1)

    first = [str(c[0]) for c in raw.columns]
    out = raw.copy()
    out.columns = first
    return out


def download_qqq_yfinance(cfg: Config, paths: Dict[str, Path]) -> pd.DataFrame:
    """Download raw QQQ data or reuse the locally cached raw CSV."""
    raw_path = paths["raw"] / f"{cfg.symbol.lower()}_yfinance.csv"

    if raw_path.exists() and not cfg.force_download:
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df.sort_index()

    yf = import_yfinance()

    kwargs = dict(
        tickers=cfg.symbol,
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        timeout=30,
    )

    # Newer yfinance versions accept multi_level_index=False. Older versions may
    # not, so use a compatibility fallback.
    try:
        raw = yf.download(**kwargs, multi_level_index=False)
    except TypeError:
        raw = yf.download(**kwargs)

    if raw is None or raw.empty:
        raise RuntimeError(
            "Downloaded QQQ data are empty. Check internet access, Yahoo/yfinance availability, "
            "and the Spyder Python environment."
        )

    raw = flatten_yfinance_columns(raw, cfg.symbol).sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw.to_csv(raw_path)

    meta = {
        "download_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": cfg.symbol,
        "source": "yfinance/Yahoo Finance",
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "rows": int(len(raw)),
        "columns": [str(x) for x in raw.columns],
        "sha256": sha256_file(raw_path),
        "yfinance_version": safe_package_version("yfinance"),
        "note": (
            "Prototype research source. Before publication, independently cross-check selected "
            "dates, adjusted prices, splits and dividends with a second data source."
        ),
    }
    (paths["metadata"] / "download_metadata_yfinance.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return raw


def download_qqq_alpha_vantage(
    api_key: str, out_path: Path, symbol: str = "QQQ"
) -> pd.DataFrame:
    """
    Optional independent cross-check using Alpha Vantage Daily Adjusted.

    NOTE: at the time this v3 file was prepared, Alpha Vantage documents Daily
    Adjusted as a premium endpoint. Therefore this function is NOT called by the
    default pipeline. It is here only for users who have appropriate access.
    """
    import requests

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "datatype": "csv",
        "apikey": api_key,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    text = response.text
    if "timestamp" not in text.lower():
        raise RuntimeError(f"Unexpected Alpha Vantage response: {text[:400]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    df = pd.read_csv(StringIO(text), parse_dates=["timestamp"])
    return df.sort_values("timestamp").set_index("timestamp")


def select_adjusted_close(raw: pd.DataFrame, require_adjusted: bool = True) -> Tuple[pd.Series, str]:
    adjusted_names = ["Adj Close", "adj_close", "adjusted_close"]
    for col in adjusted_names:
        if col in raw.columns:
            s = pd.to_numeric(raw[col], errors="coerce")
            s.name = "price"
            return s, col

    if not require_adjusted:
        for col in ["Close", "close"]:
            if col in raw.columns:
                s = pd.to_numeric(raw[col], errors="coerce")
                s.name = "price"
                return s, col

    raise KeyError(
        "Adjusted close was not found. The program refuses to silently switch reward definitions. "
        f"Available columns: {list(raw.columns)}"
    )


def validate_raw_market_data(raw: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    if raw.empty:
        raise ValueError("Raw market dataframe is empty.")

    idx = pd.to_datetime(raw.index)
    if idx.has_duplicates:
        raise ValueError("Raw market data contain duplicate dates.")
    if not idx.is_monotonic_increasing:
        raise ValueError("Raw market dates are not sorted ascending.")

    price, source_col = select_adjusted_close(raw, cfg.require_adjusted_close)
    finite = np.isfinite(price.to_numpy(dtype=float))
    positive = price.to_numpy(dtype=float) > 0

    if finite.sum() < 100:
        raise ValueError("Too few finite price observations.")
    if not np.all(positive[finite]):
        raise ValueError("Adjusted-close data contain non-positive prices.")

    return {
        "rows_raw": int(len(raw)),
        "start": str(idx.min().date()),
        "end": str(idx.max().date()),
        "price_column_used": source_col,
        "finite_price_rows": int(finite.sum()),
        "duplicate_dates": int(idx.duplicated().sum()),
    }


# =============================================================================
# 4. Leakage-safe feature construction (for the one-step RL convention)
# =============================================================================


def build_market_features(
    raw: pd.DataFrame, paths: Dict[str, Path], cfg: Config
) -> pd.DataFrame:
    quality = validate_raw_market_data(raw, cfg)
    price, price_col = select_adjusted_close(raw, cfg.require_adjusted_close)
    price = price.dropna().sort_index()

    df = pd.DataFrame(index=price.index)
    df["price"] = price

    daily_ret = price.pct_change()
    df["ret_1"] = daily_ret
    df["ret_5"] = price.pct_change(5)
    df["ret_20"] = price.pct_change(20)
    df["vol_20"] = daily_ret.rolling(20).std(ddof=1)

    ma5 = price.rolling(5).mean()
    ma20 = price.rolling(20).mean()
    df["ma_spread"] = (ma5 - ma20) / ma20

    running_peak = price.cummax()
    df["drawdown"] = price / running_peak - 1.0

    # Discrete-time research convention:
    # state/context at t -> choose one-step action -> receive t-to-t+1 reward.
    # next_return is NEVER included in MODEL_FEATURES.
    df["next_return"] = price.shift(-1) / price - 1.0

    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(df) <= cfg.min_history + cfg.oracle_horizon + max(cfg.rollout_sizes):
        raise ValueError(
            "Processed QQQ history is too short for the configured walk-forward study."
        )

    df.to_csv(paths["processed"] / "qqq_market_features.csv")

    quality.update(
        {
            "price_column_used": price_col,
            "rows_processed": int(len(df)),
            "processed_start": str(df.index.min().date()),
            "processed_end": str(df.index.max().date()),
            "model_features": MODEL_FEATURES,
            "next_return_used_as_feature": bool("next_return" in MODEL_FEATURES),
        }
    )
    (paths["metadata"] / "data_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return df


# =============================================================================
# 5. Bayesian Linear Thompson Sampling behavior policy
# =============================================================================


class BayesianLinearActionModel:
    """Bayesian linear regression with Gaussian prior and known noise variance."""

    def __init__(self, dimension: int, prior_precision: float, noise_std: float) -> None:
        self.dimension = int(dimension)
        self.noise_var = float(noise_std) ** 2
        self.precision = float(prior_precision) * np.eye(self.dimension)
        self.b = np.zeros(self.dimension, dtype=float)

    def posterior_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        # 8x8 matrices are tiny, but add a numerical fallback to avoid a fatal
        # inversion error if the online design becomes ill-conditioned.
        try:
            cov = np.linalg.inv(self.precision)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(self.precision)
        mean = cov @ self.b
        cov = (cov + cov.T) / 2.0
        return mean, cov

    def sample_theta(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        mean, cov = self.posterior_mean_cov()
        # Tiny diagonal jitter prevents numerical non-PSD warnings.
        cov = cov + 1e-12 * np.eye(self.dimension)
        return rng.multivariate_normal(mean, cov, size=n, check_valid="ignore")

    def update(self, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=float)
        inv_noise = 1.0 / max(self.noise_var, 1e-12)
        self.precision += inv_noise * np.outer(x, x)
        self.b += inv_noise * x * float(reward)


class LinearThompsonBandit:
    def __init__(self, cfg: Config, dimension: int) -> None:
        self.cfg = cfg
        self.models = {
            int(a): BayesianLinearActionModel(
                dimension=dimension,
                prior_precision=cfg.prior_precision,
                noise_std=cfg.assumed_reward_noise_std,
            )
            for a in ACTIONS
        }

    def action_probabilities(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        mc_draws: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> np.ndarray:
        draws = int(mc_draws or self.cfg.ts_mc_draws)
        eps = self.cfg.minimum_action_probability if epsilon is None else float(epsilon)

        scores = np.empty((draws, len(ACTIONS)), dtype=float)
        for j, a in enumerate(ACTIONS):
            theta = self.models[int(a)].sample_theta(rng, n=draws)
            scores[:, j] = theta @ x

        winners = np.argmax(scores, axis=1)
        counts = np.bincount(winners, minlength=len(ACTIONS)).astype(float)
        probs = counts / counts.sum()
        return apply_probability_floor(probs, eps)

    def sample_action(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        mc_draws: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        probs = self.action_probabilities(x, rng, mc_draws=mc_draws, epsilon=epsilon)
        idx = int(rng.choice(len(ACTIONS), p=probs))
        return int(ACTIONS[idx]), probs

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        self.models[int(action)].update(x, reward)


# =============================================================================
# 6. Hidden one-step oracle and simulated logged-bandit data
# =============================================================================


def context_vector(row: pd.Series, position_before: float) -> np.ndarray:
    return np.array(
        [
            1.0,
            row["ret_1"],
            row["ret_5"],
            row["ret_20"],
            row["vol_20"],
            row["ma_spread"],
            row["drawdown"],
            position_before,
        ],
        dtype=float,
    )


def oracle_rewards_for_day(
    row: pd.Series, position_before: float, cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    rewards: List[float] = []
    positions_after: List[float] = []

    for action in ACTIONS:
        position_after = clip_position(
            position_before + cfg.position_step * int(action), cfg
        )
        trade_size = abs(position_after - position_before)
        reward = (
            position_after * float(row["next_return"])
            - cfg.transaction_cost * trade_size
            - cfg.risk_lambda * (position_after**2) * (float(row["vol_20"]) ** 2)
        )
        rewards.append(float(reward))
        positions_after.append(float(position_after))

    return np.asarray(rewards), np.asarray(positions_after)


def simulate_behavior_logs(
    market: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
    epsilon: Optional[float] = None,
    mc_draws: Optional[int] = None,
) -> pd.DataFrame:
    bandit = LinearThompsonBandit(cfg, dimension=len(MODEL_FEATURES))
    position = cfg.initial_position
    records: List[Dict[str, float]] = []

    for date, row in market.iterrows():
        x = context_vector(row, position)
        oracle_rewards, positions_after = oracle_rewards_for_day(row, position, cfg)

        action, probs = bandit.sample_action(
            x, rng, mc_draws=mc_draws, epsilon=epsilon
        )
        idx = action_index(action)
        observed_reward = float(oracle_rewards[idx])
        position_after = float(positions_after[idx])

        records.append(
            {
                "date": pd.Timestamp(date),
                "intercept": 1.0,
                "ret_1": float(row["ret_1"]),
                "ret_5": float(row["ret_5"]),
                "ret_20": float(row["ret_20"]),
                "vol_20": float(row["vol_20"]),
                "ma_spread": float(row["ma_spread"]),
                "drawdown": float(row["drawdown"]),
                "position_before": float(position),
                "action": int(action),
                "behavior_prob": float(probs[idx]),
                "behavior_prob_m1": float(probs[0]),
                "behavior_prob_0": float(probs[1]),
                "behavior_prob_p1": float(probs[2]),
                "reward_observed": observed_reward,
                "reward_oracle_m1": float(oracle_rewards[0]),
                "reward_oracle_0": float(oracle_rewards[1]),
                "reward_oracle_p1": float(oracle_rewards[2]),
                "position_after": position_after,
                "next_return": float(row["next_return"]),
                "vol_20_market": float(row["vol_20"]),
            }
        )

        bandit.update(x, action, observed_reward)
        position = position_after

    logs = pd.DataFrame.from_records(records).set_index("date").sort_index()
    return logs


def run_log_sanity_checks(logs: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    required = {
        "behavior_prob",
        "behavior_prob_m1",
        "behavior_prob_0",
        "behavior_prob_p1",
        "reward_observed",
        "reward_oracle_m1",
        "reward_oracle_0",
        "reward_oracle_p1",
        "position_before",
        "position_after",
        "action",
    }
    missing = sorted(required - set(logs.columns))
    if missing:
        raise ValueError(f"Simulated log is missing columns: {missing}")

    prob_matrix = logs[["behavior_prob_m1", "behavior_prob_0", "behavior_prob_p1"]].to_numpy(float)
    sums = prob_matrix.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-9):
        raise ValueError("Behavior action probabilities do not sum to 1.")
    if np.any(prob_matrix <= 0):
        raise ValueError("Behavior propensities must be strictly positive for standard IPS/DR.")

    actions = logs["action"].to_numpy(int)
    if not set(np.unique(actions)).issubset(set(ACTIONS.tolist())):
        raise ValueError("Invalid action found in simulated log.")

    idx = observed_action_indices(logs)
    oracle_taken = oracle_reward_matrix(logs)[np.arange(len(logs)), idx]
    max_reward_mismatch = float(
        np.max(np.abs(oracle_taken - logs["reward_observed"].to_numpy(float)))
    )
    if max_reward_mismatch > 1e-10:
        raise ValueError("Observed reward does not match hidden oracle reward for chosen action.")

    pmin = float(logs["position_before"].min())
    pmax = float(logs["position_after"].max())
    if pmin < cfg.min_position - 1e-12 or pmax > cfg.max_position + 1e-12:
        raise ValueError("Position escaped configured bounds.")

    if logs[MODEL_FEATURES + ["reward_observed", "behavior_prob"]].isna().any().any():
        raise ValueError("NaN detected in core logged-bandit columns.")

    return {
        "rows": int(len(logs)),
        "probability_sum_max_abs_error": float(np.max(np.abs(sums - 1.0))),
        "minimum_any_action_probability": float(np.min(prob_matrix)),
        "minimum_observed_propensity": float(logs["behavior_prob"].min()),
        "max_reward_oracle_mismatch": max_reward_mismatch,
        "position_min": float(min(logs["position_before"].min(), logs["position_after"].min())),
        "position_max": float(max(logs["position_before"].max(), logs["position_after"].max())),
        "action_frequencies": {
            str(int(k)): float(v)
            for k, v in logs["action"].value_counts(normalize=True).sort_index().items()
        },
        "passed": True,
    }


# =============================================================================
# 7. Reward model and target policies
# =============================================================================


class RidgeRewardModel:
    """Action-specific ridge reward models with one shared feature scaler."""

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = float(alpha)
        self.scaler = StandardScaler()
        self.models: Dict[int, Ridge] = {}
        self.fallback_means: Dict[int, float] = {}
        self.is_fitted = False

    def fit(
        self, df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None
    ) -> "RidgeRewardModel":
        if len(df) == 0:
            raise ValueError("Cannot fit reward model on an empty dataframe.")

        X = model_matrix(df)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        actions = df["action"].to_numpy(dtype=int)
        y = df["reward_observed"].to_numpy(dtype=float)
        global_mean = float(np.mean(y))

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if len(sample_weight) != len(df):
                raise ValueError("Reward-model sample_weight length mismatch.")

        for a in ACTIONS:
            mask = actions == int(a)
            if np.sum(mask) >= 3:
                model = Ridge(alpha=self.alpha, fit_intercept=True)
                if sample_weight is None:
                    model.fit(Xs[mask], y[mask])
                else:
                    model.fit(Xs[mask], y[mask], sample_weight=sample_weight[mask])
                self.models[int(a)] = model
                self.fallback_means[int(a)] = float(np.mean(y[mask]))
            else:
                self.fallback_means[int(a)] = global_mean

        self.is_fitted = True
        return self

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Reward model is not fitted.")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        q = np.empty((len(Xs), len(ACTIONS)), dtype=float)
        for j, a in enumerate(ACTIONS):
            if int(a) in self.models:
                q[:, j] = self.models[int(a)].predict(Xs)
            else:
                q[:, j] = self.fallback_means[int(a)]
        return q

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_matrix(model_matrix(df))


class Policy:
    name: str = "policy"

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ScaledSoftmaxQPolicy(Policy):
    """
    Softmax target policy using a training-only q-scale.

    q_scale turns reward-unit action differences into dimensionless scores. This
    avoids target policies becoming almost uniform solely because daily rewards
    are numerically small.
    """

    def __init__(
        self,
        q_model: RidgeRewardModel,
        temperature: float,
        q_scale: float,
        min_prob: float,
        name: str,
    ) -> None:
        self.q_model = q_model
        self.temperature = float(temperature)
        self.q_scale = max(float(q_scale), 1e-12)
        self.min_prob = float(min_prob)
        self.name = name

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        q = self.q_model.predict_matrix(X)
        centered = q - np.mean(q, axis=1, keepdims=True)
        dimensionless = centered / self.q_scale
        probs = stable_softmax(dimensionless, self.temperature)
        return apply_probability_floor(probs, self.min_prob)


class UniformPolicy(Policy):
    def __init__(self, name: str = "uniform") -> None:
        self.name = name

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), len(ACTIONS)), 1.0 / len(ACTIONS), dtype=float)


class MixturePolicy(Policy):
    def __init__(
        self, first: Policy, second: Policy, second_weight: float, name: str
    ) -> None:
        self.first = first
        self.second = second
        self.second_weight = float(second_weight)
        self.name = name

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        p1 = self.first.probabilities(X)
        p2 = self.second.probabilities(X)
        eta = self.second_weight
        return (1.0 - eta) * p1 + eta * p2


def robust_q_scale(q: np.ndarray, floor: float) -> float:
    row_sd = np.std(q, axis=1, ddof=0)
    finite = row_sd[np.isfinite(row_sd)]
    if len(finite) == 0:
        return float(floor)
    return max(float(np.median(finite)), float(floor))


def fit_policy_pair(history: pd.DataFrame, cfg: Config) -> Tuple[Policy, Policy, Policy]:
    if len(history) < max(10, cfg.recent_policy_lookback):
        raise ValueError("Insufficient history to fit base/new candidate policies.")

    long_hist = history.iloc[-min(cfg.long_policy_lookback, len(history)) :]
    recent_hist = history.iloc[-min(cfg.recent_policy_lookback, len(history)) :]

    long_model = RidgeRewardModel(cfg.ridge_alpha).fit(long_hist)
    recent_model = RidgeRewardModel(cfg.ridge_alpha).fit(recent_hist)

    long_q = long_model.predict_dataframe(long_hist)
    recent_q = recent_model.predict_dataframe(recent_hist)
    long_scale = robust_q_scale(long_q, cfg.q_scale_floor)
    recent_scale = robust_q_scale(recent_q, cfg.q_scale_floor)

    base = ScaledSoftmaxQPolicy(
        long_model,
        temperature=cfg.base_temperature,
        q_scale=long_scale,
        min_prob=cfg.policy_min_prob,
        name="base_long_history",
    )
    new = ScaledSoftmaxQPolicy(
        recent_model,
        temperature=cfg.new_temperature,
        q_scale=recent_scale,
        min_prob=cfg.policy_min_prob,
        name="new_recent_history",
    )
    conservative = MixturePolicy(
        base,
        new,
        second_weight=cfg.conservative_eta,
        name="conservative_mix",
    )
    return base, new, conservative


def policy_distance_diagnostics(
    base: Policy, new: Policy, df: pd.DataFrame
) -> Dict[str, float]:
    X = model_matrix(df)
    pb = base.probabilities(X)
    pn = new.probabilities(X)
    l1 = np.sum(np.abs(pb - pn), axis=1)
    return {
        "mean_l1": float(np.mean(l1)),
        "median_l1": float(np.median(l1)),
        "max_l1": float(np.max(l1)),
        "base_mean_p_m1": float(pb[:, 0].mean()),
        "base_mean_p_0": float(pb[:, 1].mean()),
        "base_mean_p_p1": float(pb[:, 2].mean()),
        "new_mean_p_m1": float(pn[:, 0].mean()),
        "new_mean_p_0": float(pn[:, 1].mean()),
        "new_mean_p_p1": float(pn[:, 2].mean()),
    }


# =============================================================================
# 8. OPE estimators and diagnostics
# =============================================================================


@dataclass
class OPEResult:
    estimate: float
    ess: float
    ess_fraction: float
    max_importance_weight: float
    p95_importance_weight: float
    p99_importance_weight: float
    mean_importance_weight: float
    max_combined_weight_fraction: float


def target_action_probabilities(policy: Policy, df: pd.DataFrame) -> np.ndarray:
    return policy.probabilities(model_matrix(df))


def importance_weights(policy: Policy, df: pd.DataFrame) -> np.ndarray:
    probs = target_action_probabilities(policy, df)
    idx = observed_action_indices(df)
    target_taken = probs[np.arange(len(df)), idx]
    behavior = df["behavior_prob"].to_numpy(dtype=float)
    if np.any(~np.isfinite(behavior)) or np.any(behavior <= 0):
        raise ValueError("Behavior propensities must be finite and strictly positive.")
    return target_taken / behavior


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w**2))
    if denom <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / denom)


def normalize_sample_weights(sample_weights: Optional[np.ndarray], n: int) -> np.ndarray:
    if sample_weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    if len(w) != n:
        raise ValueError("sample_weights length mismatch")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or np.sum(w) <= 0:
        raise ValueError("sample_weights must be finite, nonnegative, and have positive total weight")
    return w


def ope_components(
    df: pd.DataFrame, policy: Policy, q_model: RidgeRewardModel
) -> Dict[str, np.ndarray]:
    X = model_matrix(df)
    target_probs = policy.probabilities(X)
    q_hat = q_model.predict_matrix(X)
    idx = observed_action_indices(df)
    behavior = df["behavior_prob"].to_numpy(dtype=float)
    reward = df["reward_observed"].to_numpy(dtype=float)

    if np.any(behavior <= 0):
        raise ValueError("Non-positive behavior probability in OPE data.")

    target_taken = target_probs[np.arange(len(df)), idx]
    imp = target_taken / behavior
    model_value = np.sum(target_probs * q_hat, axis=1)
    q_taken = q_hat[np.arange(len(df)), idx]
    residual = reward - q_taken
    dr = model_value + imp * residual

    return {
        "importance": imp,
        "reward": reward,
        "model_value": model_value,
        "q_taken": q_taken,
        "residual": residual,
        "dr": dr,
    }


def evaluate_ope(
    df: pd.DataFrame,
    policy: Policy,
    q_model: RidgeRewardModel,
    estimator: str,
    sample_weights: Optional[np.ndarray] = None,
) -> OPEResult:
    if len(df) == 0:
        raise ValueError("OPE dataframe is empty.")

    sw = normalize_sample_weights(sample_weights, len(df))
    comp = ope_components(df, policy, q_model)
    imp = comp["importance"]

    est = estimator.lower()
    if est == "dm":
        estimate = np.sum(sw * comp["model_value"]) / np.sum(sw)
    elif est == "ips":
        estimate = np.sum(sw * imp * comp["reward"]) / np.sum(sw)
    elif est == "snips":
        denom = np.sum(sw * imp)
        estimate = np.nan if abs(denom) < 1e-12 else np.sum(sw * imp * comp["reward"]) / denom
    elif est == "dr":
        estimate = np.sum(sw * comp["dr"]) / np.sum(sw)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    combined = sw * imp
    ess = effective_sample_size(combined)
    total_combined = float(np.sum(combined))
    max_fraction = (
        float(np.max(combined) / total_combined) if total_combined > 0 else float("nan")
    )
    return OPEResult(
        estimate=float(estimate),
        ess=ess,
        ess_fraction=float(ess / len(df)),
        max_importance_weight=float(np.max(imp)),
        p95_importance_weight=float(np.quantile(imp, 0.95)),
        p99_importance_weight=float(np.quantile(imp, 0.99)),
        mean_importance_weight=float(np.mean(imp)),
        max_combined_weight_fraction=max_fraction,
    )


def oracle_value(policy: Policy, df: pd.DataFrame) -> float:
    probs = policy.probabilities(model_matrix(df))
    rewards = oracle_reward_matrix(df)
    return float(np.mean(np.sum(probs * rewards, axis=1)))


def oracle_daily_rewards(policy: Policy, df: pd.DataFrame) -> np.ndarray:
    probs = policy.probabilities(model_matrix(df))
    rewards = oracle_reward_matrix(df)
    return np.sum(probs * rewards, axis=1)


# =============================================================================
# 9. Historical relevance: recency and regime similarity
# =============================================================================


def recency_weights(
    historical_df: pd.DataFrame, half_life_trading_days: float
) -> np.ndarray:
    """
    Exponential memory decay measured in TRADING OBSERVATIONS.

    The old v2 code used calendar-day differences while the article described
    21/63/126/252 as trading-day memories. This v3 fixes that mismatch.
    Most recent historical row has age 1, previous row age 2, etc.
    """
    n = len(historical_df)
    if n == 0:
        return np.array([], dtype=float)
    ages = np.arange(n, 0, -1, dtype=float)
    kappa = math.log(2.0) / float(half_life_trading_days)
    return np.exp(-kappa * ages)


def regime_similarity_weights(
    historical_df: pd.DataFrame, current_row: pd.Series, bandwidth: float
) -> np.ndarray:
    Z = historical_df[REGIME_FEATURES].to_numpy(dtype=float)
    current = current_row[REGIME_FEATURES].to_numpy(dtype=float)

    mean = np.mean(Z, axis=0)
    std = np.std(Z, axis=0, ddof=1)
    std = np.where((~np.isfinite(std)) | (std < 1e-8), 1.0, std)

    Zs = (Z - mean) / std
    cs = (current - mean) / std
    dist2 = np.sum((Zs - cs) ** 2, axis=1)
    h = max(float(bandwidth), 1e-8)
    return np.exp(-dist2 / (2.0 * h * h))


def combined_recency_regime_weights(
    historical_df: pd.DataFrame,
    current_row: pd.Series,
    cfg: Config,
) -> np.ndarray:
    wt = recency_weights(historical_df, cfg.half_life_trading_days)
    wr = regime_similarity_weights(historical_df, current_row, cfg.regime_bandwidth)
    return wt * wr


def historical_scheme(
    history: pd.DataFrame,
    current_row: pd.Series,
    scheme: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if scheme == "all_history":
        df = history
        w = np.ones(len(df))
    elif scheme == "last_60":
        df = history.iloc[-min(cfg.rolling_short, len(history)) :]
        w = np.ones(len(df))
    elif scheme == "last_126":
        df = history.iloc[-min(cfg.rolling_medium, len(history)) :]
        w = np.ones(len(df))
    elif scheme == "last_252":
        df = history.iloc[-min(cfg.rolling_long, len(history)) :]
        w = np.ones(len(df))
    elif scheme == "decay":
        df = history
        w = recency_weights(df, cfg.half_life_trading_days)
    elif scheme == "decay_regime":
        df = history
        w = combined_recency_regime_weights(df, current_row, cfg)
    else:
        raise ValueError(f"Unknown historical scheme: {scheme}")
    return df, w


def describe_anchor_regime(history: pd.DataFrame, current_row: pd.Series) -> str:
    """Simple descriptive regime label using only information available at the anchor."""
    vol_q75 = float(history["vol_20"].quantile(0.75))
    vol_q25 = float(history["vol_20"].quantile(0.25))
    vol = float(current_row["vol_20"])
    if vol >= vol_q75:
        vol_label = "high_vol"
    elif vol <= vol_q25:
        vol_label = "low_vol"
    else:
        vol_label = "mid_vol"
    trend_label = "up20" if float(current_row["ret_20"]) >= 0 else "down20"
    return f"{vol_label}_{trend_label}"


# =============================================================================
# 10. Recent evidence: old-policy recent history vs new-policy rollout
# =============================================================================


def simulate_current_rollout(
    current_oracle_df: pd.DataFrame,
    base_policy: Policy,
    new_policy: Policy,
    cfg: Config,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate logged feedback under (1-rho)*base + rho*new for ALL rows supplied."""
    if len(current_oracle_df) == 0:
        return current_oracle_df.copy()

    source = current_oracle_df.copy()
    X = model_matrix(source)
    base_probs = base_policy.probabilities(X)
    new_probs = new_policy.probabilities(X)

    rho = cfg.rollout_new_policy_mix
    behavior_probs_all = (1.0 - rho) * base_probs + rho * new_probs
    rewards = oracle_reward_matrix(source)

    sampled_actions: List[int] = []
    sampled_probs: List[float] = []
    observed_rewards: List[float] = []

    for i in range(len(source)):
        idx = int(rng.choice(len(ACTIONS), p=behavior_probs_all[i]))
        sampled_actions.append(int(ACTIONS[idx]))
        sampled_probs.append(float(behavior_probs_all[i, idx]))
        observed_rewards.append(float(rewards[i, idx]))

    source["action"] = sampled_actions
    source["behavior_prob"] = sampled_probs
    source["behavior_prob_m1"] = behavior_probs_all[:, 0]
    source["behavior_prob_0"] = behavior_probs_all[:, 1]
    source["behavior_prob_p1"] = behavior_probs_all[:, 2]
    source["reward_observed"] = observed_rewards
    source["rollout_observation"] = 1
    return source


def take_most_recent(df: pd.DataFrame, m: int) -> pd.DataFrame:
    if m <= 0:
        return df.iloc[:0].copy()
    return df.iloc[-min(int(m), len(df)) :].copy()


# =============================================================================
# 11. Moving-block bootstrap for paired Delta V
# =============================================================================


def moving_block_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    L = max(1, min(int(block_length), n))
    indices: List[int] = []
    while len(indices) < n:
        start = int(rng.integers(0, n - L + 1))
        indices.extend(range(start, start + L))
    return np.asarray(indices[:n], dtype=int)


def weighted_mean(
    values: np.ndarray, weights: np.ndarray, idx: Optional[np.ndarray] = None
) -> float:
    if idx is not None:
        values = values[idx]
        weights = weights[idx]
    denom = float(np.sum(weights))
    if denom <= 0:
        raise ValueError("Weighted mean has non-positive total weight.")
    return float(np.sum(weights * values) / denom)


def bootstrap_policy_delta_lcb(
    df: pd.DataFrame,
    new_policy: Policy,
    base_policy: Policy,
    q_model: RidgeRewardModel,
    sample_weights: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> Dict[str, float]:
    new_comp = ope_components(df, new_policy, q_model)
    base_comp = ope_components(df, base_policy, q_model)

    point_new = weighted_mean(new_comp["dr"], sample_weights)
    point_base = weighted_mean(base_comp["dr"], sample_weights)
    delta = point_new - point_base

    boot = np.empty(cfg.bootstrap_reps, dtype=float)
    for b in range(cfg.bootstrap_reps):
        idx = moving_block_indices(len(df), cfg.bootstrap_block_length, rng)
        boot[b] = (
            weighted_mean(new_comp["dr"], sample_weights, idx)
            - weighted_mean(base_comp["dr"], sample_weights, idx)
        )

    se = float(np.std(boot, ddof=1))
    z = float(norm.ppf(cfg.confidence_level))
    lcb_normal = float(delta - z * se)
    alpha = 1.0 - cfg.confidence_level
    lcb_percentile = float(np.quantile(boot, alpha))

    return {
        "estimate_new": float(point_new),
        "estimate_base": float(point_base),
        "delta": float(delta),
        "se_delta": se,
        "lcb_normal": lcb_normal,
        "lcb_percentile": lcb_percentile,
        "bootstrap_q05": float(np.quantile(boot, 0.05)),
        "bootstrap_q50": float(np.quantile(boot, 0.50)),
        "bootstrap_q95": float(np.quantile(boot, 0.95)),
    }


# =============================================================================
# 12. Walk-forward anchor definitions and validation/test split
# =============================================================================


def candidate_anchors(logs: pd.DataFrame, cfg: Config, reserve_days: int = 0) -> List[int]:
    last_start = len(logs) - cfg.oracle_horizon - reserve_days
    if last_start <= cfg.min_history:
        return []
    return list(range(cfg.min_history, last_start, cfg.anchor_step))


def split_anchors_chronologically(
    anchors: List[int], cfg: Config
) -> Tuple[List[int], List[int]]:
    if len(anchors) < 4:
        return anchors, []
    cut = max(1, min(len(anchors) - 1, int(len(anchors) * cfg.validation_fraction)))
    return anchors[:cut], anchors[cut:]


# =============================================================================
# 13. Experiment 1: standard OPE accuracy on held-out logged windows
# =============================================================================


def experiment_static_ope(
    logs: pd.DataFrame, cfg: Config, anchors: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Evaluate OPE on multiple held-out PRE-anchor windows.

    For each anchor i:
      policy/reward-model training: logs before the evaluation window
      OPE + oracle comparison: the immediately following held-out eval window

    This is cleaner than using one final 252-day window and gives enough repeated
    windows to compute ranking diagnostics.
    """
    if anchors is None:
        anchors = candidate_anchors(logs, cfg)

    rows: List[Dict[str, object]] = []
    for i in anchors:
        eval_start = i - cfg.static_eval_window
        if eval_start <= cfg.recent_policy_lookback:
            continue
        train_df = logs.iloc[:eval_start].copy()
        eval_df = logs.iloc[eval_start:i].copy()
        if len(eval_df) < max(30, cfg.static_eval_window // 2):
            continue

        base, new, conservative = fit_policy_pair(train_df, cfg)
        uniform = UniformPolicy()
        candidates = [base, new, conservative, uniform]

        q_train = train_df.iloc[-min(cfg.long_policy_lookback, len(train_df)) :]
        q_model = RidgeRewardModel(cfg.ridge_alpha).fit(q_train)

        for policy in candidates:
            truth = oracle_value(policy, eval_df)
            for estimator in ["dm", "ips", "snips", "dr"]:
                res = evaluate_ope(eval_df, policy, q_model, estimator)
                rows.append(
                    {
                        "anchor_date": pd.Timestamp(logs.index[i]),
                        "eval_start": pd.Timestamp(eval_df.index[0]),
                        "eval_end": pd.Timestamp(eval_df.index[-1]),
                        "policy": policy.name,
                        "estimator": estimator.upper(),
                        "estimate": res.estimate,
                        "oracle": truth,
                        "absolute_error": abs(res.estimate - truth),
                        "signed_error": res.estimate - truth,
                        "ess": res.ess,
                        "ess_fraction": res.ess_fraction,
                        "max_importance_weight": res.max_importance_weight,
                        "p99_importance_weight": res.p99_importance_weight,
                    }
                )
    return pd.DataFrame(rows)


def summarize_static(static_df: pd.DataFrame) -> pd.DataFrame:
    if static_df.empty:
        return pd.DataFrame()

    base_summary = (
        static_df.groupby("estimator", as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("signed_error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            Bias=("signed_error", "mean"),
            Mean_ESS=("ess", "mean"),
            Median_ESS=("ess", "median"),
            Mean_ESS_Fraction=("ess_fraction", "mean"),
            Max_Weight=("max_importance_weight", "max"),
            N_Comparisons=("absolute_error", "size"),
        )
    )

    ranking_rows = []
    for estimator, est_df in static_df.groupby("estimator"):
        cors: List[float] = []
        top_correct: List[float] = []
        for _, g in est_df.groupby("anchor_date"):
            if g["policy"].nunique() < 2:
                continue
            rho = spearmanr(g["oracle"], g["estimate"]).statistic
            if np.isfinite(rho):
                cors.append(float(rho))
            oracle_best = g.loc[g["oracle"].idxmax(), "policy"]
            estimated_best = g.loc[g["estimate"].idxmax(), "policy"]
            top_correct.append(float(oracle_best == estimated_best))
        ranking_rows.append(
            {
                "estimator": estimator,
                "Mean_Spearman_Rank": float(np.mean(cors)) if cors else np.nan,
                "Best_Policy_Selection_Accuracy": float(np.mean(top_correct)) if top_correct else np.nan,
                "N_Windows": int(est_df["anchor_date"].nunique()),
            }
        )

    ranking = pd.DataFrame(ranking_rows)
    return base_summary.merge(ranking, on="estimator", how="left").sort_values("MAE")


# =============================================================================
# 14. Experiment 2: historical-memory schemes under nonstationarity
# =============================================================================


def experiment_nonstationary_memory(
    logs: pd.DataFrame, cfg: Config, anchors: Optional[List[int]] = None
) -> pd.DataFrame:
    schemes = [
        "all_history",
        "last_60",
        "last_126",
        "last_252",
        "decay",
        "decay_regime",
    ]
    anchors = anchors or candidate_anchors(logs, cfg)
    rows: List[Dict[str, object]] = []

    for i in anchors:
        history = logs.iloc[:i].copy()
        current_row = logs.iloc[i]
        future = logs.iloc[i : i + cfg.oracle_horizon].copy()
        if len(future) < cfg.oracle_horizon:
            continue

        _, new, _ = fit_policy_pair(history, cfg)
        truth = oracle_value(new, future)
        q_train = history.iloc[-min(cfg.long_policy_lookback, len(history)) :]
        q_model = RidgeRewardModel(cfg.ridge_alpha).fit(q_train)
        regime_label = describe_anchor_regime(history, current_row)

        for scheme in schemes:
            ope_df, sw = historical_scheme(history, current_row, scheme, cfg)
            res = evaluate_ope(ope_df, new, q_model, "dr", sample_weights=sw)
            rows.append(
                {
                    "anchor_date": pd.Timestamp(logs.index[i]),
                    "regime_label": regime_label,
                    "scheme": scheme,
                    "estimate": res.estimate,
                    "oracle_future": truth,
                    "absolute_error": abs(res.estimate - truth),
                    "signed_error": res.estimate - truth,
                    "ess": res.ess,
                    "ess_fraction": res.ess_fraction,
                    "n_nominal": len(ope_df),
                    "max_importance_weight": res.max_importance_weight,
                }
            )
    return pd.DataFrame(rows)


def summarize_nonstationary(nonstat_df: pd.DataFrame) -> pd.DataFrame:
    if nonstat_df.empty:
        return pd.DataFrame()
    return (
        nonstat_df.groupby("scheme", as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("signed_error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            Bias=("signed_error", "mean"),
            Mean_ESS=("ess", "mean"),
            Median_ESS=("ess", "median"),
            Mean_ESS_Fraction=("ess_fraction", "mean"),
            N_Windows=("anchor_date", "nunique"),
        )
        .sort_values("MAE")
    )


def summarize_nonstationary_by_regime(nonstat_df: pd.DataFrame) -> pd.DataFrame:
    if nonstat_df.empty:
        return pd.DataFrame()
    return (
        nonstat_df.groupby(["regime_label", "scheme"], as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            Bias=("signed_error", "mean"),
            Mean_ESS=("ess", "mean"),
            N_Windows=("anchor_date", "nunique"),
        )
        .sort_values(["regime_label", "MAE"])
    )


# =============================================================================
# 15. Experiment 3: recent history vs active new-policy rollout
# =============================================================================


def experiment_recent_evidence(
    logs: pd.DataFrame,
    cfg: Config,
    anchors: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    At every anchor, reserve a fixed max-rollout period immediately before one
    common future oracle window. Compare:

    * no recent evidence (m=0)
    * most recent m observations under the ORIGINAL behavior log -> recency
    * the same most recent m dates re-logged under base/new mixture -> coverage

    The rollout action sequence is simulated ONCE per anchor and nested, so m=20
    really contains the same most-recent evidence as m=10 plus 10 more rows. This
    removes the v2 confound where each m was a fresh random rollout.
    """
    max_rollout = max(cfg.rollout_sizes)
    anchors = anchors or candidate_anchors(logs, cfg, reserve_days=max_rollout)
    rows: List[Dict[str, object]] = []

    for i in anchors:
        history = logs.iloc[:i].copy()
        pool = logs.iloc[i : i + max_rollout].copy()
        future = logs.iloc[i + max_rollout : i + max_rollout + cfg.oracle_horizon].copy()
        if len(pool) < max_rollout or len(future) < cfg.oracle_horizon:
            continue

        # Current relevance is assessed at the end of the evidence-collection
        # period, immediately before the future oracle window.
        current_row = future.iloc[0]
        base, new, _ = fit_policy_pair(history, cfg)
        truth = oracle_value(new, future)
        hist_weights = combined_recency_regime_weights(history, current_row, cfg)

        q_train = history.iloc[-min(cfg.long_policy_lookback, len(history)) :]
        q_model = RidgeRewardModel(cfg.ridge_alpha).fit(q_train)

        full_rollout = simulate_current_rollout(
            pool,
            base,
            new,
            cfg,
            deterministic_rng(cfg.seed, i, salt=301),
        )

        for m in cfg.rollout_sizes:
            m = int(m)
            if m == 0:
                combined = history.copy()
                weights = hist_weights.copy()
                res = evaluate_ope(combined, new, q_model, "dr", sample_weights=weights)
                rows.append(
                    {
                        "anchor_date": pd.Timestamp(future.index[0]),
                        "m_evidence": 0,
                        "evidence_type": "history_only",
                        "estimate": res.estimate,
                        "oracle_future": truth,
                        "absolute_error": abs(res.estimate - truth),
                        "signed_error": res.estimate - truth,
                        "ess": res.ess,
                        "ess_fraction": res.ess_fraction,
                        "n_history": len(history),
                        "n_recent": 0,
                    }
                )
                continue

            recent_old = take_most_recent(pool, m)
            recent_rollout = take_most_recent(full_rollout, m)

            for evidence_type, recent_df in [
                ("recent_old_policy", recent_old),
                ("new_policy_rollout", recent_rollout),
            ]:
                combined = pd.concat([history, recent_df], axis=0)
                weights = np.concatenate([hist_weights, np.ones(len(recent_df))])
                res = evaluate_ope(combined, new, q_model, "dr", sample_weights=weights)
                rows.append(
                    {
                        "anchor_date": pd.Timestamp(future.index[0]),
                        "m_evidence": m,
                        "evidence_type": evidence_type,
                        "estimate": res.estimate,
                        "oracle_future": truth,
                        "absolute_error": abs(res.estimate - truth),
                        "signed_error": res.estimate - truth,
                        "ess": res.ess,
                        "ess_fraction": res.ess_fraction,
                        "n_history": len(history),
                        "n_recent": len(recent_df),
                    }
                )
    return pd.DataFrame(rows)


def summarize_recent_evidence(evidence_df: pd.DataFrame) -> pd.DataFrame:
    if evidence_df.empty:
        return pd.DataFrame()
    return (
        evidence_df.groupby(["evidence_type", "m_evidence"], as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            Median_Error=("absolute_error", "median"),
            Bias=("signed_error", "mean"),
            Mean_ESS=("ess", "mean"),
            Mean_ESS_Fraction=("ess_fraction", "mean"),
            N_Windows=("anchor_date", "nunique"),
        )
        .sort_values(["evidence_type", "m_evidence"])
    )


# =============================================================================
# 16. Validation: select memory / rollout hyperparameters, then freeze
# =============================================================================


def validation_parameter_grid(
    logs: pd.DataFrame, cfg: Config
) -> Tuple[pd.DataFrame, Dict[str, float], List[int], List[int]]:
    max_rollout = max(cfg.rollout_sizes)
    anchors = candidate_anchors(logs, cfg, reserve_days=max_rollout)
    validation_anchors, test_anchors = split_anchors_chronologically(anchors, cfg)

    if not validation_anchors or not test_anchors:
        raise ValueError(
            "Not enough chronological anchors to form both validation and test sets. "
            "Reduce min_history/oracle_horizon only for debugging, not for final publication."
        )

    rows: List[Dict[str, object]] = []
    for half_life in cfg.half_life_grid:
        for bandwidth in cfg.regime_bandwidth_grid:
            trial_cfg = replace(
                cfg,
                half_life_trading_days=int(half_life),
                regime_bandwidth=float(bandwidth),
            )
            ev = experiment_recent_evidence(logs, trial_cfg, anchors=validation_anchors)
            roll = ev[ev["evidence_type"] == "new_policy_rollout"].copy()
            if roll.empty:
                continue
            for m, group in roll.groupby("m_evidence"):
                rows.append(
                    {
                        "half_life_trading_days": int(half_life),
                        "regime_bandwidth": float(bandwidth),
                        "m_rollout": int(m),
                        "validation_mae": float(group["absolute_error"].mean()),
                        "validation_bias": float(group["signed_error"].mean()),
                        "validation_mean_ess": float(group["ess"].mean()),
                        "validation_mean_ess_fraction": float(group["ess_fraction"].mean()),
                        "n_validation_anchors": int(group["anchor_date"].nunique()),
                    }
                )

    grid = pd.DataFrame(rows)
    if grid.empty:
        raise RuntimeError("Validation grid produced no results.")

    # Prefer configurations with reasonable mean ESS when possible; among them,
    # minimize validation MAE. ESS is not optimized on the final test data.
    eligible = grid[grid["validation_mean_ess"] >= cfg.minimum_ess]
    pool = eligible if not eligible.empty else grid
    ordered = pool.sort_values(
        ["validation_mae", "validation_mean_ess"], ascending=[True, False]
    )
    best = ordered.iloc[0]
    chosen = {
        "half_life_trading_days": float(best["half_life_trading_days"]),
        "regime_bandwidth": float(best["regime_bandwidth"]),
        "m_rollout": float(best["m_rollout"]),
    }
    return grid, chosen, validation_anchors, test_anchors


def validation_ess_fraction_gate_grid(
    logs: pd.DataFrame,
    cfg: Config,
    validation_anchors: List[int],
    m_rollout: int,
) -> Tuple[pd.DataFrame, float]:
    """
    Select a normalized ESS gate using VALIDATION anchors only.

    The grid is evaluated with the Point+ESS decision rule because ESS is a data-
    coverage diagnostic, not a confidence-level tuning knob. A threshold is
    eligible only if it still permits a minimum validation adoption rate. Among
    eligible thresholds we prefer higher selection accuracy, then lower unsafe
    rate, then higher adoption rate. If no threshold is eligible, use 0.0 rather
    than inventing a test-driven gate.
    """
    rows: List[Dict[str, object]] = []
    for frac in cfg.ess_fraction_gate_grid:
        trial_cfg = replace(
            cfg,
            minimum_ess_fraction=float(frac),
            bootstrap_reps=min(cfg.bootstrap_reps, cfg.validation_bootstrap_reps),
        )
        decisions, _ = experiment_safe_update(
            logs, trial_cfg, test_anchors=validation_anchors, m_rollout=m_rollout
        )
        s = summarize_safe(decisions)
        if s.empty:
            continue
        r = s.iloc[0]
        unsafe = r["point_ess_unsafe_rate"]
        rows.append(
            {
                "minimum_ess_fraction": float(frac),
                "selection_accuracy": float(r["point_ess_selection_accuracy"]),
                "unsafe_rate": float(unsafe) if np.isfinite(unsafe) else np.nan,
                "n_adoptions": int(r["point_ess_n_adoptions"]),
                "adoption_rate": float(r["point_ess_adoption_rate"]),
                "false_rejection_rate": float(r["point_ess_false_rejection_rate"]),
                "gate_pass_rate": float(r["ess_gate_pass_rate"]),
                "n_validation_windows": int(r["n_test_windows"]),
            }
        )

    grid = pd.DataFrame(rows).sort_values("minimum_ess_fraction")
    if grid.empty:
        return grid, 0.0

    eligible = grid[grid["adoption_rate"] >= cfg.minimum_validation_adoption_rate].copy()
    pool = eligible if not eligible.empty else grid.copy()
    pool["unsafe_for_sort"] = pool["unsafe_rate"].fillna(1.0)
    pool = pool.sort_values(
        ["selection_accuracy", "unsafe_for_sort", "adoption_rate", "minimum_ess_fraction"],
        ascending=[False, True, False, True],
    )
    chosen = float(pool.iloc[0]["minimum_ess_fraction"])
    return grid, chosen


# =============================================================================
# 17. Experiment 4: Delta V, LCB, ESS, and conservative policy improvement
# =============================================================================


def experiment_safe_update(
    logs: pd.DataFrame,
    cfg: Config,
    test_anchors: List[int],
    m_rollout: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    path_rows: List[Dict[str, object]] = []

    for i in test_anchors:
        history = logs.iloc[:i].copy()
        pool = logs.iloc[i : i + m_rollout].copy()
        future = logs.iloc[i + m_rollout : i + m_rollout + cfg.oracle_horizon].copy()
        if len(pool) < m_rollout or len(future) < cfg.oracle_horizon:
            continue

        current_row = future.iloc[0]
        base, new, _ = fit_policy_pair(history, cfg)
        rollout = simulate_current_rollout(
            pool,
            base,
            new,
            cfg,
            deterministic_rng(cfg.seed, i, salt=401),
        )

        hist_weights = combined_recency_regime_weights(history, current_row, cfg)
        combined = pd.concat([history, rollout], axis=0) if len(rollout) else history.copy()
        weights = (
            np.concatenate([hist_weights, np.ones(len(rollout))])
            if len(rollout)
            else hist_weights.copy()
        )

        q_train = history.iloc[-min(cfg.long_policy_lookback, len(history)) :]
        q_model = RidgeRewardModel(cfg.ridge_alpha).fit(q_train)

        delta_stats = bootstrap_policy_delta_lcb(
            combined,
            new,
            base,
            q_model,
            weights,
            cfg,
            deterministic_rng(cfg.seed, i, salt=402),
        )
        new_ope = evaluate_ope(combined, new, q_model, "dr", sample_weights=weights)
        base_ope = evaluate_ope(combined, base, q_model, "dr", sample_weights=weights)

        oracle_new = oracle_value(new, future)
        oracle_base = oracle_value(base, future)
        true_delta = oracle_new - oracle_base
        true_new_better = bool(true_delta > 0)

        ess_absolute_pass = bool(
            new_ope.ess >= cfg.minimum_ess and base_ope.ess >= cfg.minimum_ess
        )
        ess_fraction_pass = bool(
            new_ope.ess_fraction >= cfg.minimum_ess_fraction
            and base_ope.ess_fraction >= cfg.minimum_ess_fraction
        )
        ess_pass = bool(ess_absolute_pass and ess_fraction_pass)
        point_adopt = bool(delta_stats["delta"] > 0)
        point_ess_adopt = bool(point_adopt and ess_pass)
        lcb_ess_adopt = bool(delta_stats["lcb_normal"] > 0 and ess_pass)

        deployed = (
            MixturePolicy(
                base,
                new,
                second_weight=cfg.conservative_eta,
                name="safe_mixture",
            )
            if lcb_ess_adopt
            else base
        )

        rows.append(
            {
                "anchor_date": pd.Timestamp(future.index[0]),
                "estimate_new": new_ope.estimate,
                "estimate_base": base_ope.estimate,
                "delta_hat": delta_stats["delta"],
                "se_delta": delta_stats["se_delta"],
                "lcb_normal": delta_stats["lcb_normal"],
                "lcb_percentile": delta_stats["lcb_percentile"],
                "bootstrap_q05": delta_stats["bootstrap_q05"],
                "bootstrap_q50": delta_stats["bootstrap_q50"],
                "bootstrap_q95": delta_stats["bootstrap_q95"],
                "ess_new": new_ope.ess,
                "ess_base": base_ope.ess,
                "ess_fraction_new": new_ope.ess_fraction,
                "ess_fraction_base": base_ope.ess_fraction,
                "ess_absolute_pass": int(ess_absolute_pass),
                "ess_fraction_pass": int(ess_fraction_pass),
                "minimum_ess_fraction_used": float(cfg.minimum_ess_fraction),
                "ess_gate_pass": int(ess_pass),
                "oracle_new": oracle_new,
                "oracle_base": oracle_base,
                "true_delta": true_delta,
                "true_new_better": int(true_new_better),
                "adopt_point": int(point_adopt),
                "adopt_point_ess": int(point_ess_adopt),
                "adopt_lcb_ess": int(lcb_ess_adopt),
                "unsafe_point": int(point_adopt and not true_new_better),
                "unsafe_point_ess": int(point_ess_adopt and not true_new_better),
                "unsafe_lcb_ess": int(lcb_ess_adopt and not true_new_better),
            }
        )

        deployed_daily = oracle_daily_rewards(deployed, future)
        base_daily = oracle_daily_rewards(base, future)
        new_daily = oracle_daily_rewards(new, future)
        for j, date in enumerate(future.index):
            path_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "anchor_date": pd.Timestamp(future.index[0]),
                    "deployed_reward": float(deployed_daily[j]),
                    "base_reward": float(base_daily[j]),
                    "new_reward": float(new_daily[j]),
                    "adopt_lcb_ess": int(lcb_ess_adopt),
                }
            )

    decisions = pd.DataFrame(rows)
    path = pd.DataFrame(path_rows)
    if not path.empty:
        path = path.sort_values(["date", "anchor_date"])
    return decisions, path


def _unsafe_rate(unsafe_mask: pd.Series, adopt_mask: pd.Series) -> float:
    denom = int(adopt_mask.sum())
    # IMPORTANT: 0 adoptions means unsafe rate is undefined, NOT 0%.
    return float(unsafe_mask.sum() / denom) if denom > 0 else float("nan")


def summarize_safe(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()

    truth = decisions["true_new_better"] == 1
    rules = {
        "point": decisions["adopt_point"] == 1,
        "point_ess": decisions["adopt_point_ess"] == 1,
        "lcb_ess": decisions["adopt_lcb_ess"] == 1,
    }

    out: Dict[str, object] = {
        "n_test_windows": int(len(decisions)),
        "true_new_better_rate": float(truth.mean()),
        "ess_gate_pass_rate": float(decisions["ess_gate_pass"].mean()),
        "ess_absolute_pass_rate": float(decisions["ess_absolute_pass"].mean()),
        "ess_fraction_pass_rate": float(decisions["ess_fraction_pass"].mean()),
        "minimum_ess_fraction_used": float(decisions["minimum_ess_fraction_used"].iloc[0]),
        "mean_lcb_normal": float(decisions["lcb_normal"].mean()),
        "mean_ess_new": float(decisions["ess_new"].mean()),
        "mean_ess_base": float(decisions["ess_base"].mean()),
        "mean_ess_fraction_new": float(decisions["ess_fraction_new"].mean()),
        "mean_ess_fraction_base": float(decisions["ess_fraction_base"].mean()),
    }

    for name, adopt in rules.items():
        unsafe = adopt & (~truth)
        out[f"{name}_selection_accuracy"] = float((adopt == truth).mean())
        out[f"{name}_unsafe_rate"] = _unsafe_rate(unsafe, adopt)
        out[f"{name}_n_adoptions"] = int(adopt.sum())
        out[f"{name}_adoption_rate"] = float(adopt.mean())
        # Among truly better candidates, how often was adoption missed?
        denom_better = int(truth.sum())
        out[f"{name}_false_rejection_rate"] = (
            float(((~adopt) & truth).sum() / denom_better) if denom_better > 0 else np.nan
        )

    return pd.DataFrame([out])


# =============================================================================
# 18. Experiment 5: exploration / coverage / ESS / behavior-performance tradeoff
# =============================================================================


def action_entropy(freqs: np.ndarray) -> float:
    p = np.asarray(freqs, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def experiment_exploration_sweep(
    market: pd.DataFrame,
    cfg: Config,
    fixed_target_policy: Policy,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold the TARGET policy fixed while changing behavior-policy exploration.

    v2 refit the target separately for each epsilon, which confounded the effect
    of logging coverage with a changing target. v3 removes that confound.
    """
    rows: List[Dict[str, object]] = []

    for repeat in range(cfg.exploration_repeats):
        # Use the same repeat seed for every epsilon. Different probabilities will
        # still lead to different actions, but this is closer to common-random-
        # numbers than giving every epsilon an unrelated seed.
        repeat_seed = cfg.seed + 20_000 + repeat

        for eps in cfg.exploration_epsilons:
            rng = np.random.default_rng(repeat_seed)
            logs = simulate_behavior_logs(
                market,
                cfg,
                rng,
                epsilon=float(eps),
                mc_draws=cfg.exploration_ts_mc_draws,
            )
            eval_df = logs.iloc[-cfg.static_eval_window :].copy()
            train_df = logs.iloc[: -cfg.static_eval_window].copy()
            q_train = train_df.iloc[-min(cfg.long_policy_lookback, len(train_df)) :]
            q_model = RidgeRewardModel(cfg.ridge_alpha).fit(q_train)

            ope = evaluate_ope(eval_df, fixed_target_policy, q_model, "dr")
            truth = oracle_value(fixed_target_policy, eval_df)

            freq_series = eval_df["action"].value_counts(normalize=True)
            freqs = np.array(
                [
                    float(freq_series.get(-1, 0.0)),
                    float(freq_series.get(0, 0.0)),
                    float(freq_series.get(1, 0.0)),
                ]
            )
            behavior_rewards = eval_df["reward_observed"].to_numpy(float)
            behavior_wealth = wealth_and_drawdown(behavior_rewards)

            rows.append(
                {
                    "repeat": repeat,
                    "epsilon": float(eps),
                    "ope_estimate": ope.estimate,
                    "oracle": truth,
                    "absolute_error": abs(ope.estimate - truth),
                    "ess": ope.ess,
                    "ess_fraction": ope.ess_fraction,
                    "freq_action_m1": freqs[0],
                    "freq_action_0": freqs[1],
                    "freq_action_p1": freqs[2],
                    "minimum_action_frequency": float(np.min(freqs)),
                    "action_entropy": action_entropy(freqs),
                    "behavior_mean_reward": float(np.mean(behavior_rewards)),
                    "behavior_final_synthetic_wealth": float(behavior_wealth["wealth"].iloc[-1]),
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("epsilon", as_index=False)
        .agg(
            OPE_MAE=("absolute_error", "mean"),
            OPE_Error_SD=("absolute_error", "std"),
            Mean_ESS=("ess", "mean"),
            ESS_SD=("ess", "std"),
            Mean_ESS_Fraction=("ess_fraction", "mean"),
            Mean_Min_Action_Freq=("minimum_action_frequency", "mean"),
            Mean_Action_Entropy=("action_entropy", "mean"),
            Mean_Behavior_Reward=("behavior_mean_reward", "mean"),
            Mean_Behavior_Final_Wealth=("behavior_final_synthetic_wealth", "mean"),
            Freq_M1=("freq_action_m1", "mean"),
            Freq_0=("freq_action_0", "mean"),
            Freq_P1=("freq_action_p1", "mean"),
            N_Repeats=("repeat", "nunique"),
        )
        .sort_values("epsilon")
    )
    return detail, summary


# =============================================================================
# 19. Article figures: save to disk AND show in Spyder
# =============================================================================


def save_market_figures(logs: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    plt.figure(figsize=(10, 5))
    wealth = np.cumprod(1.0 + logs["next_return"].to_numpy(dtype=float))
    plt.plot(logs.index, wealth)
    plt.xlabel("Date")
    plt.ylabel("QQQ adjusted-price cumulative growth (normalized)")
    plt.title("F2a: QQQ market path used as the experimental environment")
    finalize_figure(fig_dir / "F2a_qqq_market_path.png", cfg)

    plt.figure(figsize=(10, 5))
    plt.plot(logs.index, logs["vol_20"], label="20-day volatility")
    plt.plot(logs.index, logs["drawdown"], label="drawdown")
    plt.xlabel("Date")
    plt.title("F2b: Nonstationary market context")
    plt.legend()
    finalize_figure(fig_dir / "F2b_volatility_drawdown.png", cfg)


def plot_static_ope(static_df: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if static_df.empty:
        return
    plt.figure(figsize=(8, 6))
    for estimator, group in static_df.groupby("estimator"):
        plt.scatter(group["oracle"], group["estimate"], label=estimator, s=30, alpha=0.7)
    lo = min(static_df["oracle"].min(), static_df["estimate"].min())
    hi = max(static_df["oracle"].max(), static_df["estimate"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Hidden oracle policy value")
    plt.ylabel("OPE estimate")
    plt.title("F3: Can standard OPE recover the hidden truth?")
    plt.legend()
    finalize_figure(fig_dir / "F3_ope_vs_oracle.png", cfg)


def plot_ess_vs_ope_error(static_df: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    """Figure for the article claim: ESS is useful, but not sufficient."""
    if static_df.empty:
        return
    plt.figure(figsize=(8, 5))
    for estimator in ["IPS", "SNIPS", "DR"]:
        g = static_df[static_df["estimator"] == estimator].copy()
        if g.empty:
            continue
        rho = spearmanr(g["ess_fraction"], g["absolute_error"]).statistic
        label = f"{estimator} (Spearman={rho:.2f})" if np.isfinite(rho) else estimator
        plt.scatter(g["ess_fraction"], g["absolute_error"], alpha=0.55, s=24, label=label)
    plt.xlabel("ESS / nominal sample size")
    plt.ylabel("Absolute OPE error")
    plt.title("F3b: Effective sample fraction vs OPE error")
    plt.legend()
    finalize_figure(fig_dir / "F3b_ess_fraction_vs_ope_error.png", cfg)


def plot_nonstationary(summary: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.bar(summary["scheme"], summary["MAE"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean absolute OPE-to-future-oracle error")
    plt.title("F4: Historical-memory choice vs value-transfer error")
    finalize_figure(fig_dir / "F4_memory_scheme_mae.png", cfg)


def plot_recent_evidence(summary: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(8, 5))
    for evidence_type, g in summary.groupby("evidence_type"):
        plt.plot(g["m_evidence"], g["MAE"], marker="o", label=evidence_type)
    plt.xlabel("Number of recent observations m")
    plt.ylabel("Mean absolute OPE error")
    plt.title("F5: Recent old-policy data vs new-policy rollout")
    plt.legend()
    finalize_figure(fig_dir / "F5_recent_evidence_vs_error.png", cfg)


def plot_recent_evidence_ess(summary: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(8, 5))
    for evidence_type, g in summary.groupby("evidence_type"):
        plt.plot(g["m_evidence"], g["Mean_ESS"], marker="o", label=evidence_type)
    plt.xlabel("Number of recent observations m")
    plt.ylabel("Mean effective sample size")
    plt.title("F5b: Recent evidence changes coverage differently")
    plt.legend()
    finalize_figure(fig_dir / "F5b_recent_evidence_vs_ess.png", cfg)


def plot_weight_diagnostics(
    logs: pd.DataFrame, cfg: Config, fig_dir: Path
) -> None:
    if len(logs) <= cfg.static_eval_window + cfg.recent_policy_lookback:
        return
    train_df = logs.iloc[: -cfg.static_eval_window]
    eval_df = logs.iloc[-cfg.static_eval_window :]
    _, target, _ = fit_policy_pair(train_df, cfg)
    imp = importance_weights(target, eval_df)
    ess = effective_sample_size(imp)

    plt.figure(figsize=(8, 5))
    plt.hist(imp, bins=40)
    plt.xlabel("Importance weight")
    plt.ylabel("Count")
    plt.title(f"F6: Importance weights (ESS={ess:.1f}, N={len(imp)})")
    finalize_figure(fig_dir / "F6_importance_weights.png", cfg)


def plot_safe_update(
    decisions: pd.DataFrame, path: pd.DataFrame, fig_dir: Path, cfg: Config
) -> None:
    if decisions.empty:
        return

    s = summarize_safe(decisions)
    if not s.empty:
        r = s.iloc[0]
        labels = ["Point", "Point+ESS", "LCB+ESS"]
        vals = [
            r["point_selection_accuracy"],
            r["point_ess_selection_accuracy"],
            r["lcb_ess_selection_accuracy"],
        ]
        plt.figure(figsize=(8, 5))
        plt.bar(labels, vals)
        plt.ylim(0, 1)
        plt.ylabel("Selection accuracy")
        plt.title("F7a: Correct policy-update decisions")
        finalize_figure(fig_dir / "F7a_selection_accuracy.png", cfg)

        unsafe_vals = [
            r["point_unsafe_rate"],
            r["point_ess_unsafe_rate"],
            r["lcb_ess_unsafe_rate"],
        ]
        plt.figure(figsize=(8, 5))
        # matplotlib can display NaN bars as blank, which correctly signals an
        # undefined unsafe rate when that rule never adopted a new policy.
        plt.bar(labels, unsafe_vals)
        plt.ylim(0, 1)
        plt.ylabel("Unsafe rate among adoptions")
        plt.title("F7b: Unsafe adoption rate")
        finalize_figure(fig_dir / "F7b_unsafe_rate.png", cfg)

    if not path.empty:
        # Overlapping future windows make this an auxiliary visualization only.
        grouped = path.groupby("date", as_index=False).agg(
            deployed_reward=("deployed_reward", "mean"),
            base_reward=("base_reward", "mean"),
            new_reward=("new_reward", "mean"),
        )
        safe_w = wealth_and_drawdown(grouped["deployed_reward"].to_numpy())
        base_w = wealth_and_drawdown(grouped["base_reward"].to_numpy())
        new_w = wealth_and_drawdown(grouped["new_reward"].to_numpy())

        plt.figure(figsize=(10, 5))
        plt.plot(grouped["date"], safe_w["wealth"], label="LCB-gated deployed")
        plt.plot(grouped["date"], base_w["wealth"], label="Base policy")
        plt.plot(grouped["date"], new_w["wealth"], label="Always-new policy")
        plt.xlabel("Date")
        plt.ylabel("Synthetic expected wealth")
        plt.title("F8a: Auxiliary paper-trading path")
        plt.legend()
        finalize_figure(fig_dir / "F8a_safe_update_wealth.png", cfg)

        plt.figure(figsize=(10, 5))
        plt.plot(grouped["date"], safe_w["drawdown"], label="LCB-gated deployed")
        plt.plot(grouped["date"], base_w["drawdown"], label="Base policy")
        plt.plot(grouped["date"], new_w["drawdown"], label="Always-new policy")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.title("F8b: Auxiliary drawdown path")
        plt.legend()
        finalize_figure(fig_dir / "F8b_safe_update_drawdown.png", cfg)


def plot_exploration(summary: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if summary.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(summary["epsilon"], summary["Mean_ESS"], marker="o")
    plt.xlabel("Minimum exploration probability epsilon")
    plt.ylabel("Mean ESS")
    plt.title("F9a: Exploration floor vs OPE effective sample size")
    finalize_figure(fig_dir / "F9a_exploration_vs_ess.png", cfg)

    plt.figure(figsize=(8, 5))
    plt.plot(summary["epsilon"], summary["OPE_MAE"], marker="o")
    plt.xlabel("Minimum exploration probability epsilon")
    plt.ylabel("OPE MAE")
    plt.title("F9b: Exploration floor vs OPE error")
    finalize_figure(fig_dir / "F9b_exploration_vs_ope_error.png", cfg)

    plt.figure(figsize=(8, 5))
    plt.plot(summary["epsilon"], summary["Mean_Behavior_Reward"], marker="o")
    plt.xlabel("Minimum exploration probability epsilon")
    plt.ylabel("Mean behavior reward in evaluation window")
    plt.title("F9c: Exploration vs behavior performance")
    finalize_figure(fig_dir / "F9c_exploration_vs_behavior_reward.png", cfg)


def plot_validation_grid(grid: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if grid.empty:
        return
    best_by_m = grid.groupby("m_rollout", as_index=False)["validation_mae"].min()
    plt.figure(figsize=(8, 5))
    plt.plot(best_by_m["m_rollout"], best_by_m["validation_mae"], marker="o")
    plt.xlabel("New-policy rollout observations m")
    plt.ylabel("Best validation MAE across memory settings")
    plt.title("F10: Validation-only hyperparameter selection")
    finalize_figure(fig_dir / "F10_validation_grid.png", cfg)


def plot_ess_gate_validation(grid: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if grid.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(grid["minimum_ess_fraction"], grid["selection_accuracy"], marker="o", label="Selection accuracy")
    plt.plot(grid["minimum_ess_fraction"], grid["adoption_rate"], marker="o", label="Adoption rate")
    plt.xlabel("Minimum ESS fraction gate")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.title("F10b: Validation-only ESS-fraction gate selection")
    plt.legend()
    finalize_figure(fig_dir / "F10b_ess_fraction_gate_validation.png", cfg)


# =============================================================================
# 20. Publication robustness: independent master behavior seeds
# =============================================================================


def run_master_seed_robustness(
    market: pd.DataFrame, cfg: Config
) -> Dict[str, pd.DataFrame]:
    """
    Repeat Experiments 1-4 after regenerating the entire Thompson-sampling
    historical behavior log under independent master seeds.

    These outputs do NOT replace the detailed main-seed results. They answer the
    publication question: are the qualitative findings stable to the stochastic
    historical logging process?
    """
    exp1_rows: List[pd.DataFrame] = []
    exp2_rows: List[pd.DataFrame] = []
    exp3_rows: List[pd.DataFrame] = []
    exp4_rows: List[pd.DataFrame] = []
    chosen_rows: List[Dict[str, object]] = []

    for seed in cfg.master_behavior_seeds:
        seed_cfg = replace(cfg, seed=int(seed), show_figures_in_spyder=False)
        logs = simulate_behavior_logs(market, seed_cfg, set_seed(int(seed)))
        run_log_sanity_checks(logs, seed_cfg)

        grid, chosen, val_anchors, test_anchors = validation_parameter_grid(logs, seed_cfg)
        locked = replace(
            seed_cfg,
            half_life_trading_days=int(chosen["half_life_trading_days"]),
            regime_bandwidth=float(chosen["regime_bandwidth"]),
        )
        locked_m = int(chosen["m_rollout"])
        ess_grid, ess_frac = validation_ess_fraction_gate_grid(
            logs, locked, val_anchors, locked_m
        )
        locked = replace(locked, minimum_ess_fraction=float(ess_frac))
        chosen_rows.append(
            {
                "seed": int(seed),
                "half_life_trading_days": int(chosen["half_life_trading_days"]),
                "regime_bandwidth": float(chosen["regime_bandwidth"]),
                "m_rollout": locked_m,
                "minimum_ess_fraction": float(ess_frac),
                "n_validation_anchors": len(val_anchors),
                "n_test_anchors": len(test_anchors),
            }
        )

        e1 = summarize_static(experiment_static_ope(logs, locked, test_anchors))
        if not e1.empty:
            e1.insert(0, "seed", int(seed))
            exp1_rows.append(e1)

        e2 = summarize_nonstationary(
            experiment_nonstationary_memory(logs, locked, test_anchors)
        )
        if not e2.empty:
            e2.insert(0, "seed", int(seed))
            exp2_rows.append(e2)

        e3 = summarize_recent_evidence(
            experiment_recent_evidence(logs, locked, test_anchors)
        )
        if not e3.empty:
            e3.insert(0, "seed", int(seed))
            exp3_rows.append(e3)

        decisions, _ = experiment_safe_update(logs, locked, test_anchors, locked_m)
        e4 = summarize_safe(decisions)
        if not e4.empty:
            e4.insert(0, "seed", int(seed))
            exp4_rows.append(e4)

    return {
        "chosen": pd.DataFrame(chosen_rows),
        "exp01": pd.concat(exp1_rows, ignore_index=True) if exp1_rows else pd.DataFrame(),
        "exp02": pd.concat(exp2_rows, ignore_index=True) if exp2_rows else pd.DataFrame(),
        "exp03": pd.concat(exp3_rows, ignore_index=True) if exp3_rows else pd.DataFrame(),
        "exp04": pd.concat(exp4_rows, ignore_index=True) if exp4_rows else pd.DataFrame(),
    }


def summarize_master_seed_robustness(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compact cross-seed table used in the Medium robustness paragraph."""
    rows: List[Dict[str, object]] = []

    e1 = results.get("exp01", pd.DataFrame())
    if not e1.empty:
        for estimator, g in e1.groupby("estimator"):
            rows.append({
                "experiment": "Exp1_OPE",
                "group": str(estimator),
                "metric": "MAE",
                "mean": float(g["MAE"].mean()),
                "sd": float(g["MAE"].std(ddof=1)) if len(g) > 1 else np.nan,
                "n_seeds": int(g["seed"].nunique()),
            })

    e2 = results.get("exp02", pd.DataFrame())
    if not e2.empty:
        for scheme, g in e2.groupby("scheme"):
            rows.append({
                "experiment": "Exp2_Memory",
                "group": str(scheme),
                "metric": "MAE",
                "mean": float(g["MAE"].mean()),
                "sd": float(g["MAE"].std(ddof=1)) if len(g) > 1 else np.nan,
                "n_seeds": int(g["seed"].nunique()),
            })

    e3 = results.get("exp03", pd.DataFrame())
    if not e3.empty:
        for (etype, m), g in e3.groupby(["evidence_type", "m_evidence"]):
            rows.append({
                "experiment": "Exp3_RecentEvidence",
                "group": f"{etype}:m={int(m)}",
                "metric": "MAE",
                "mean": float(g["MAE"].mean()),
                "sd": float(g["MAE"].std(ddof=1)) if len(g) > 1 else np.nan,
                "n_seeds": int(g["seed"].nunique()),
            })

    e4 = results.get("exp04", pd.DataFrame())
    if not e4.empty:
        for metric in [
            "point_selection_accuracy",
            "point_unsafe_rate",
            "lcb_ess_selection_accuracy",
            "lcb_ess_adoption_rate",
            "lcb_ess_unsafe_rate",
        ]:
            vals = pd.to_numeric(e4[metric], errors="coerce")
            rows.append({
                "experiment": "Exp4_SafeUpdate",
                "group": "all",
                "metric": metric,
                "mean": float(vals.mean()) if vals.notna().any() else np.nan,
                "sd": float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan,
                "n_seeds": int(e4["seed"].nunique()),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 21. Supplementary controlled LCB stress test
# =============================================================================


def _simulate_ar1_noise(n: int, std: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    phi = float(np.clip(phi, -0.95, 0.95))
    innovation_std = float(std) * math.sqrt(max(1.0 - phi * phi, 1e-12))
    out = np.zeros(n, dtype=float)
    out[0] = rng.normal(0.0, std)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + rng.normal(0.0, innovation_std)
    return out


def synthetic_lcb_stress_test(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Controlled diagnostic for the conservative decision gate.

    We observe a correlated sequence of paired new-minus-base value differences
    whose TRUE mean Delta-V is known. We then compare a point-estimate adoption
    rule with the same one-sided moving-block-bootstrap LCB logic used in the QQQ
    experiment. This is deliberately supplementary: it validates gate behavior,
    not trading performance and not OPE estimation itself.
    """
    rows: List[Dict[str, object]] = []
    z = float(norm.ppf(cfg.confidence_level))
    n = int(cfg.synthetic_safety_n)

    for delta_idx, true_delta in enumerate(cfg.synthetic_safety_deltas):
        for rep in range(cfg.synthetic_safety_repeats):
            rng = deterministic_rng(cfg.seed + 900_000, rep, salt=delta_idx + 1)
            noise = _simulate_ar1_noise(
                n, cfg.synthetic_safety_noise_std, cfg.synthetic_safety_ar1, rng
            )
            paired = float(true_delta) + noise
            delta_hat = float(np.mean(paired))

            boot = np.empty(cfg.bootstrap_reps, dtype=float)
            for b in range(cfg.bootstrap_reps):
                idx = moving_block_indices(n, cfg.bootstrap_block_length, rng)
                boot[b] = float(np.mean(paired[idx]))
            se = float(np.std(boot, ddof=1))
            lcb = float(delta_hat - z * se)
            point_adopt = bool(delta_hat > 0)
            lcb_adopt = bool(lcb > 0)
            truly_better = bool(true_delta > 0)

            rows.append({
                "true_delta": float(true_delta),
                "repeat": int(rep),
                "delta_hat": delta_hat,
                "se_delta": se,
                "lcb": lcb,
                "point_adopt": int(point_adopt),
                "lcb_adopt": int(lcb_adopt),
                "true_better": int(truly_better),
                "point_correct": int(point_adopt == truly_better),
                "lcb_correct": int(lcb_adopt == truly_better),
                "point_unsafe": int(point_adopt and not truly_better),
                "lcb_unsafe": int(lcb_adopt and not truly_better),
            })

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("true_delta", as_index=False)
        .agg(
            Point_Adoption_Rate=("point_adopt", "mean"),
            LCB_Adoption_Rate=("lcb_adopt", "mean"),
            Point_Selection_Accuracy=("point_correct", "mean"),
            LCB_Selection_Accuracy=("lcb_correct", "mean"),
            Point_Unsafe_Rate_Per_Trial=("point_unsafe", "mean"),
            LCB_Unsafe_Rate_Per_Trial=("lcb_unsafe", "mean"),
            Mean_SE=("se_delta", "mean"),
            N_Repeats=("repeat", "nunique"),
        )
        .sort_values("true_delta")
    )
    return detail, summary


def plot_synthetic_lcb_stress(summary: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(summary["true_delta"], summary["Point_Adoption_Rate"], marker="o", label="Point estimate")
    plt.plot(summary["true_delta"], summary["LCB_Adoption_Rate"], marker="o", label="LCB")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("True paired Delta-V")
    plt.ylabel("Adoption probability")
    plt.ylim(-0.02, 1.02)
    plt.title("F11: Controlled safety-gate stress test")
    plt.legend()
    finalize_figure(fig_dir / "F11_synthetic_lcb_gate_power.png", cfg)


# =============================================================================
# 22. Automatic console/report summaries for the Medium article
# =============================================================================


def print_descriptive_conclusions(
    reporter: Reporter,
    static_summary: pd.DataFrame,
    nonstat_summary: pd.DataFrame,
    evidence_summary: pd.DataFrame,
    safe_summary: pd.DataFrame,
    exploration_summary: pd.DataFrame,
) -> None:
    reporter.section("AUTOMATIC DESCRIPTIVE OBSERVATIONS (NOT publication claims until verified)")

    if not static_summary.empty:
        best = static_summary.sort_values("MAE").iloc[0]
        reporter.write(
            f"Experiment 1: lowest observed MAE = {best['estimator']} "
            f"({best['MAE']:.6g}); mean rank correlation = "
            f"{best['Mean_Spearman_Rank']:.3f}."
        )

    if not nonstat_summary.empty:
        best = nonstat_summary.sort_values("MAE").iloc[0]
        reporter.write(
            f"Experiment 2: lowest observed test MAE memory scheme = {best['scheme']} "
            f"({best['MAE']:.6g}); this is descriptive, not a universal optimum."
        )

    if not evidence_summary.empty:
        roll = evidence_summary[evidence_summary["evidence_type"] == "new_policy_rollout"]
        old = evidence_summary[evidence_summary["evidence_type"] == "recent_old_policy"]
        if not roll.empty:
            best = roll.sort_values("MAE").iloc[0]
            reporter.write(
                f"Experiment 3: best observed new-policy-rollout m = {int(best['m_evidence'])}, "
                f"MAE={best['MAE']:.6g}."
            )
        if not roll.empty and not old.empty:
            common = sorted(set(roll["m_evidence"]) & set(old["m_evidence"]))
            if common:
                m = common[-1]
                rmae = float(roll.loc[roll["m_evidence"] == m, "MAE"].iloc[0])
                omae = float(old.loc[old["m_evidence"] == m, "MAE"].iloc[0])
                reporter.write(
                    f"At m={m}, new-policy rollout MAE={rmae:.6g} vs recent old-policy "
                    f"data MAE={omae:.6g}; use this comparison to discuss coverage vs recency."
                )

    if not safe_summary.empty:
        r = safe_summary.iloc[0]
        reporter.write(
            "Experiment 4: point vs LCB+ESS -- "
            f"SelectionAccuracy {r['point_selection_accuracy']:.3f} -> "
            f"{r['lcb_ess_selection_accuracy']:.3f}; "
            f"UnsafeRate {r['point_unsafe_rate']} -> {r['lcb_ess_unsafe_rate']}; "
            f"adoptions {int(r['point_n_adoptions'])} -> {int(r['lcb_ess_n_adoptions'])}."
        )
        reporter.write(
            "Remember: an UnsafeRate of NaN means the rule made zero adoptions; it must NOT be "
            "reported as 0%."
        )

    if not exploration_summary.empty:
        best_ess = exploration_summary.sort_values("Mean_ESS", ascending=False).iloc[0]
        best_err = exploration_summary.sort_values("OPE_MAE").iloc[0]
        reporter.write(
            f"Experiment 5: maximum mean ESS occurred at epsilon={best_ess['epsilon']:.3f}; "
            f"minimum mean OPE error occurred at epsilon={best_err['epsilon']:.3f}."
        )
        reporter.write(
            "Compare those rows with Mean_Behavior_Reward before claiming a learning/evaluation "
            "trade-off."
        )
        reporter.write(
            "v3.1 note: Experiment 5 now uses many repeats; final prose should report mean and "
            "dispersion rather than a single stochastic logging run."
        )


# =============================================================================
# 23. Results-template writer
# =============================================================================


def write_results_template(paths: Dict[str, Path], chosen: Dict[str, float]) -> None:
    text = "# Medium Article Results Template\n\n"
    text += "This file remains a draft until numerical outputs are independently verified.\n\n"
    text += "## Locked validation choices\n"
    text += f"- half_life_trading_days: {int(chosen['half_life_trading_days'])}\n"
    text += f"- regime_bandwidth: {chosen['regime_bandwidth']}\n"
    text += f"- m_rollout: {int(chosen['m_rollout'])}\n"
    if "minimum_ess_fraction" in chosen:
        text += f"- minimum_ess_fraction: {chosen['minimum_ess_fraction']}\n"
    text += "\n"
    text += "## Result 1 - Standard OPE vs hidden oracle\n[TBD]\n\n"
    text += "## Result 2 - Historical memory under nonstationarity\n[TBD]\n\n"
    text += "## Result 3 - Recent history vs new-policy rollout\n[TBD]\n\n"
    text += "## Result 4 - Delta V, LCB, ESS and unsafe deployment\n[TBD]\n\n"
    text += "## Result 5 - Exploration / coverage / ESS / behavior trade-off\n[TBD]\n\n"
    text += "## Robustness across independent behavior seeds\n[TBD]\n\n"
    text += "## Supplementary controlled LCB stress test\n[TBD]\n\n"
    text += "## Robustness and limitations\n[TBD]\n"
    (paths["results"] / "RESULTS_TEMPLATE.md").write_text(text, encoding="utf-8")


# =============================================================================
# 24. Main pipeline
# =============================================================================


def main(cfg: Optional[Config] = None) -> None:
    cfg = cfg or Config()
    paths = ensure_project_dirs(cfg)
    save_environment_metadata(cfg, paths)
    reporter = Reporter(paths["logs"] / "SPYDER_CONSOLE_REPORT.txt")

    try:
        reporter.section("QQQ CONTEXTUAL BANDIT + OPE v3.1")
        reporter.write(f"Output root: {paths['root'].resolve()}")
        reporter.write(f"Python: {sys.version.split()[0]}")
        reporter.write(f"yfinance: {safe_package_version('yfinance')}")

        reporter.section("[1/13] Download / load QQQ data")
        raw = download_qqq_yfinance(cfg, paths)
        reporter.write(f"Raw rows: {len(raw):,}")

        reporter.section("[2/13] Build market features")
        market = build_market_features(raw, paths, cfg)
        reporter.write(
            f"Processed rows: {len(market):,}; period {market.index.min().date()} -> "
            f"{market.index.max().date()}"
        )
        reporter.write(f"MODEL_FEATURES = {MODEL_FEATURES}")
        reporter.write("Confirmed: next_return is reward-only and is NOT in MODEL_FEATURES.")

        reporter.section("[3/13] Simulate Thompson-sampling historical behavior log")
        logs = simulate_behavior_logs(market, cfg, set_seed(cfg.seed))
        logs.to_csv(paths["processed"] / "qqq_simulated_bandit_logs.csv")
        checks = run_log_sanity_checks(logs, cfg)
        (paths["metadata"] / "log_sanity_checks.json").write_text(
            json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        reporter.write(json.dumps(checks, indent=2, ensure_ascii=False))
        save_market_figures(logs, paths["figures"], cfg)

        # Build validation/test anchors before running article results.
        reporter.section("[4/13] Validation-only memory / rollout selection")
        grid, chosen, validation_anchors, test_anchors = validation_parameter_grid(logs, cfg)
        grid.to_csv(paths["tables"] / "validation_parameter_grid.csv", index=False)
        reporter.write(f"Validation anchors: {len(validation_anchors)}")
        reporter.write(f"Untouched test anchors: {len(test_anchors)}")
        plot_validation_grid(grid, paths["figures"], cfg)

        locked_cfg = replace(
            cfg,
            half_life_trading_days=int(chosen["half_life_trading_days"]),
            regime_bandwidth=float(chosen["regime_bandwidth"]),
        )
        locked_m = int(chosen["m_rollout"])

        # v3.1: select an ESS/N gate on validation anchors only, then freeze it.
        ess_gate_grid, chosen_ess_fraction = validation_ess_fraction_gate_grid(
            logs, locked_cfg, validation_anchors, locked_m
        )
        ess_gate_grid.to_csv(paths["tables"] / "validation_ess_fraction_gate.csv", index=False)
        plot_ess_gate_validation(ess_gate_grid, paths["figures"], cfg)
        chosen["minimum_ess_fraction"] = float(chosen_ess_fraction)
        locked_cfg = replace(locked_cfg, minimum_ess_fraction=float(chosen_ess_fraction))
        (paths["metadata"] / "chosen_parameters.json").write_text(
            json.dumps(chosen, indent=2), encoding="utf-8"
        )
        reporter.write(f"LOCKED choices: {chosen}")

        # Candidate-policy distance check. If base/new are almost identical,
        # OPE is too easy and the article cannot make a meaningful comparison.
        latest_train = logs.iloc[: -locked_cfg.static_eval_window]
        base_check, new_check, _ = fit_policy_pair(latest_train, locked_cfg)
        dist = policy_distance_diagnostics(
            base_check, new_check, logs.iloc[-locked_cfg.static_eval_window :]
        )
        (paths["metadata"] / "candidate_policy_distance.json").write_text(
            json.dumps(dist, indent=2), encoding="utf-8"
        )
        reporter.write(f"Candidate-policy distance diagnostics: {dist}")
        if dist["mean_l1"] < locked_cfg.minimum_mean_policy_l1_distance:
            reporter.write(
                "WARNING: base/new policies are very similar. Consider revising policy temperatures "
                "or candidate construction before interpreting OPE as a challenging off-policy test."
            )

        reporter.section("[5/13] Experiment 1 - standard OPE vs hidden oracle")
        static_df = experiment_static_ope(logs, locked_cfg, anchors=test_anchors)
        static_summary = summarize_static(static_df)
        static_df.to_csv(paths["tables"] / "exp01_ope_detail.csv", index=False)
        static_summary.to_csv(paths["tables"] / "exp01_ope_summary.csv", index=False)
        reporter.dataframe(static_summary, "Experiment 1 summary:")
        plot_static_ope(static_df, paths["figures"], locked_cfg)
        plot_ess_vs_ope_error(static_df, paths["figures"], locked_cfg)
        plot_weight_diagnostics(logs, locked_cfg, paths["figures"])

        reporter.section("[6/13] Experiment 2 - historical memory on untouched test anchors")
        nonstat_df = experiment_nonstationary_memory(logs, locked_cfg, anchors=test_anchors)
        nonstat_summary = summarize_nonstationary(nonstat_df)
        nonstat_regime = summarize_nonstationary_by_regime(nonstat_df)
        nonstat_df.to_csv(paths["tables"] / "exp02_memory_detail.csv", index=False)
        nonstat_summary.to_csv(paths["tables"] / "exp02_memory_summary.csv", index=False)
        nonstat_regime.to_csv(paths["tables"] / "exp02_memory_by_regime.csv", index=False)
        reporter.dataframe(nonstat_summary, "Experiment 2 overall summary:")
        reporter.dataframe(nonstat_regime, "Experiment 2 by descriptive regime:")
        plot_nonstationary(nonstat_summary, paths["figures"], locked_cfg)

        reporter.section("[7/13] Experiment 3 - recent history vs new-policy rollout")
        evidence_df = experiment_recent_evidence(logs, locked_cfg, anchors=test_anchors)
        evidence_summary = summarize_recent_evidence(evidence_df)
        evidence_df.to_csv(paths["tables"] / "exp03_recent_evidence_detail.csv", index=False)
        evidence_summary.to_csv(paths["tables"] / "exp03_recent_evidence_summary.csv", index=False)
        reporter.dataframe(evidence_summary, "Experiment 3 summary:")
        plot_recent_evidence(evidence_summary, paths["figures"], locked_cfg)
        plot_recent_evidence_ess(evidence_summary, paths["figures"], locked_cfg)

        reporter.section("[8/13] Experiment 4 - Delta V + moving-block bootstrap + LCB + ESS gate")
        decisions, path = experiment_safe_update(
            logs, locked_cfg, test_anchors=test_anchors, m_rollout=locked_m
        )
        safe_summary = summarize_safe(decisions)
        decisions.to_csv(paths["tables"] / "exp04_safe_update_decisions.csv", index=False)
        path.to_csv(paths["tables"] / "exp04_safe_update_path.csv", index=False)
        safe_summary.to_csv(paths["tables"] / "exp04_safe_update_summary.csv", index=False)
        reporter.dataframe(safe_summary, "Experiment 4 summary:")
        plot_safe_update(decisions, path, paths["figures"], locked_cfg)

        reporter.section("[9/13] Experiment 5 - exploration / coverage / ESS trade-off")
        # Hold target policy fixed across epsilons so only behavior-data collection changes.
        fixed_train = logs.iloc[: -locked_cfg.static_eval_window]
        _, fixed_target, _ = fit_policy_pair(fixed_train, locked_cfg)
        exploration_detail, exploration_summary = experiment_exploration_sweep(
            market, locked_cfg, fixed_target
        )
        exploration_detail.to_csv(paths["tables"] / "exp05_exploration_detail.csv", index=False)
        exploration_summary.to_csv(paths["tables"] / "exp05_exploration_summary.csv", index=False)
        reporter.dataframe(exploration_summary, "Experiment 5 summary:")
        plot_exploration(exploration_summary, paths["figures"], locked_cfg)

        reporter.section("[10/13] Robustness - independent master behavior seeds")
        robustness_compact = pd.DataFrame()
        if locked_cfg.run_master_seed_robustness:
            robust = run_master_seed_robustness(market, locked_cfg)
            robust["chosen"].to_csv(paths["tables"] / "robustness_chosen_parameters_by_seed.csv", index=False)
            robust["exp01"].to_csv(paths["tables"] / "robustness_exp01_by_seed.csv", index=False)
            robust["exp02"].to_csv(paths["tables"] / "robustness_exp02_by_seed.csv", index=False)
            robust["exp03"].to_csv(paths["tables"] / "robustness_exp03_by_seed.csv", index=False)
            robust["exp04"].to_csv(paths["tables"] / "robustness_exp04_by_seed.csv", index=False)
            robustness_compact = summarize_master_seed_robustness(robust)
            robustness_compact.to_csv(paths["tables"] / "robustness_summary_compact.csv", index=False)
            reporter.dataframe(robust["chosen"], "Locked choices by master seed:")
            reporter.dataframe(robustness_compact, "Compact robustness summary:")
        else:
            reporter.write("Master-seed robustness disabled in Config.")

        reporter.section("[11/13] Supplementary controlled LCB stress test")
        if locked_cfg.run_synthetic_safety_stress_test:
            synth_detail, synth_summary = synthetic_lcb_stress_test(locked_cfg)
            synth_detail.to_csv(paths["tables"] / "supp_synthetic_lcb_stress_detail.csv", index=False)
            synth_summary.to_csv(paths["tables"] / "supp_synthetic_lcb_stress_summary.csv", index=False)
            reporter.dataframe(synth_summary, "Synthetic safety-gate summary:")
            plot_synthetic_lcb_stress(synth_summary, paths["figures"], locked_cfg)
        else:
            reporter.write("Synthetic safety stress test disabled in Config.")

        reporter.section("[12/13] Descriptive article-ready observations")
        print_descriptive_conclusions(
            reporter,
            static_summary,
            nonstat_summary,
            evidence_summary,
            safe_summary,
            exploration_summary,
        )

        reporter.section("[13/13] Save article result template and final locations")
        write_results_template(paths, chosen)
        reporter.write("All numerical tables are CSV files; all article figures are PNG files.")
        reporter.write(f"Tables:  {paths['tables'].resolve()}")
        reporter.write(f"Figures: {paths['figures'].resolve()}")
        reporter.write(f"Metadata: {paths['metadata'].resolve()}")
        reporter.write(f"Console report: {reporter.path.resolve()}")
        reporter.write(
            "IMPORTANT: these are empirical outputs from a simulated one-step RL laboratory. "
            "Verify them before writing final Medium conclusions."
        )

    except Exception as exc:
        reporter.section("FATAL ERROR")
        reporter.write(f"{type(exc).__name__}: {exc}")
        reporter.write(
            "The message above was also saved to SPYDER_CONSOLE_REPORT.txt. Fix the cause and rerun."
        )
        reporter.save()
        raise
    else:
        reporter.save()


# =============================================================================
# 25. Optional synthetic smoke test (does not download data)
# =============================================================================


def synthetic_smoke_test() -> None:
    """
    Fast end-to-end logic check for debugging. This is NOT an article experiment.
    It deliberately uses a temporary directory and shorter windows.
    """
    import tempfile

    rng = np.random.default_rng(12345)
    n = 900
    dates = pd.bdate_range("2019-01-01", periods=n)
    returns = np.concatenate(
        [
            rng.normal(0.0004, 0.010, n // 2),
            rng.normal(-0.0001, 0.018, n - n // 2),
        ]
    )
    price = 100.0 * np.cumprod(1.0 + returns)
    raw = pd.DataFrame({"Adj Close": price}, index=dates)

    with tempfile.TemporaryDirectory() as td:
        cfg = Config(
            project_dir=td,
            min_history=220,
            anchor_step=60,
            oracle_horizon=40,
            static_eval_window=80,
            long_policy_lookback=140,
            recent_policy_lookback=50,
            rollout_sizes=(0, 5, 10, 20),
            half_life_grid=(20, 60),
            regime_bandwidth_grid=(0.5, 1.0),
            validation_fraction=0.50,
            bootstrap_reps=40,
            bootstrap_block_length=5,
            ts_mc_draws=30,
            exploration_ts_mc_draws=20,
            exploration_epsilons=(0.01, 0.05),
            exploration_repeats=2,
            minimum_ess=5,
            ess_fraction_gate_grid=(0.0, 0.02),
            validation_bootstrap_reps=20,
            run_master_seed_robustness=False,
            run_synthetic_safety_stress_test=False,
            show_figures_in_spyder=False,
        )
        paths = ensure_project_dirs(cfg)
        market = build_market_features(raw, paths, cfg)
        logs = simulate_behavior_logs(market, cfg, set_seed(cfg.seed))
        run_log_sanity_checks(logs, cfg)
        grid, chosen, validation_anchors, test_anchors = validation_parameter_grid(logs, cfg)
        locked = replace(
            cfg,
            half_life_trading_days=int(chosen["half_life_trading_days"]),
            regime_bandwidth=float(chosen["regime_bandwidth"]),
        )
        ess_grid, ess_frac = validation_ess_fraction_gate_grid(
            logs, locked, validation_anchors, int(chosen["m_rollout"])
        )
        assert not ess_grid.empty
        locked = replace(locked, minimum_ess_fraction=float(ess_frac))
        static = experiment_static_ope(logs, locked, test_anchors)
        assert not summarize_static(static).empty
        nonstat = experiment_nonstationary_memory(logs, locked, test_anchors)
        assert not summarize_nonstationary(nonstat).empty
        ev = experiment_recent_evidence(logs, locked, test_anchors)
        assert not summarize_recent_evidence(ev).empty
        dec, _ = experiment_safe_update(logs, locked, test_anchors, int(chosen["m_rollout"]))
        assert not summarize_safe(dec).empty
        _, target, _ = fit_policy_pair(logs.iloc[:-locked.static_eval_window], locked)
        _, exs = experiment_exploration_sweep(market, locked, target)
        assert not exs.empty
        assert not grid.empty

    print("Synthetic smoke test PASSED.")


if __name__ == "__main__":
    # Normal Spyder run: execute the real QQQ pipeline.
    main()

    # For a quick offline logic check instead, temporarily comment main() above
    # and uncomment the next line:
    # synthetic_smoke_test()
