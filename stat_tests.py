from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED_COLUMNS = [
    "process_step",
    "batch_id",
    "batch_status",
    "batch_yield",
]


@dataclass
class TestConfig:
    alpha: float = 0.05
    normality_alpha: float = 0.05
    min_normality_n: int = 8
    adjustment: str = "holm"
    maximize_outcome: bool = True


def compare_good_bad_by_step(
    df: pd.DataFrame,
    config: Optional[TestConfig] = None,
) -> pd.DataFrame:
    """
    Compare good vs bad yields per process_step using an auto-selected test.

    Returns a DataFrame with one row per process_step and columns:
    process_step, test_result, p_value, rank.

    If a step lacks sufficient data, test_result is "insufficient_data" and
    p_value is NaN.
    """
    config = config or TestConfig()
    _validate_columns(df)

    clean_df = df.copy()
    clean_df["batch_status"] = (
        clean_df["batch_status"].astype(str).str.strip().str.lower()
    )
    clean_df["batch_yield"] = pd.to_numeric(
        clean_df["batch_yield"], errors="coerce"
    )
    clean_df = clean_df.dropna(subset=["batch_status", "batch_yield"])

    rows = []
    for step, step_df in clean_df.groupby("process_step", sort=False):
        test_df = step_df[step_df["batch_status"].isin(["good", "bad"])]
        good = test_df.loc[test_df["batch_status"] == "good", "batch_yield"].to_numpy()
        bad = test_df.loc[test_df["batch_status"] == "bad", "batch_yield"].to_numpy()

        test_result, p_value = _compare_groups_auto(
            good,
            bad,
            alpha=config.alpha,
            normality_alpha=config.normality_alpha,
            min_normality_n=config.min_normality_n,
        )

        rows.append(
            {
                "process_step": step,
                "test_result": test_result,
                "p_value": p_value,
            }
        )

    result = pd.DataFrame(rows)
    result["rank"] = (
        result["p_value"].rank(method="min", ascending=True).astype(int)
    )
    result = result.sort_values("rank", kind="stable").reset_index(drop=True)
    return result


def make_dummy_data(
    total_batches: int = 150,
    steps: Optional[List[str]] = None,
    statuses: Optional[List[str]] = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Create dummy data for quick manual testing."""
    rng = np.random.default_rng(seed)
    steps = steps or ["etch", "rinse", "polish"]
    statuses = statuses or ["good", "bad"]

    data = []
    step_count = len(steps)
    status_count = len(statuses)
    base_per_step = total_batches // step_count
    step_remainder = total_batches % step_count

    for step_idx, step in enumerate(steps):
        per_step = base_per_step + (1 if step_idx < step_remainder else 0)
        base_per_status = per_step // status_count
        status_remainder = per_step % status_count

        for status_idx, status in enumerate(statuses):
            per_status = base_per_status + (1 if status_idx < status_remainder else 0)
            for i in range(per_status):
                mean, sd = _status_yield_params(status)
                data.append(
                    {
                        "process_step": step,
                        "batch_id": f"{step}-{status}-{i:03d}",
                        "batch_status": status,
                        "batch_yield": float(rng.normal(mean, sd)),
                    }
                )

    return pd.DataFrame(data)


def _status_yield_params(status: str) -> Tuple[float, float]:
    status = status.strip().lower()
    if status == "good":
        return 92.0, 3.0
    if status == "bad":
        return 85.0, 4.0
    return 88.0, 3.5


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _compare_groups_auto(
    good: np.ndarray,
    bad: np.ndarray,
    *,
    alpha: float,
    normality_alpha: float,
    min_normality_n: int,
) -> Tuple[str, float]:
    if good.size < 2 or bad.size < 2:
        return "insufficient_data", float("nan")

    use_ttest = False
    if good.size >= min_normality_n and bad.size >= min_normality_n:
        good_p = stats.shapiro(good).pvalue
        bad_p = stats.shapiro(bad).pvalue
        use_ttest = (good_p > normality_alpha) and (bad_p > normality_alpha)

    if use_ttest:
        p_value = float(
            stats.ttest_ind(good, bad, equal_var=False, nan_policy="omit").pvalue
        )
    else:
        p_value = float(
            stats.mannwhitneyu(good, bad, alternative="two-sided").pvalue
        )

    test_result = (
        "statistically different" if p_value < alpha else "not statistically different"
    )
    return test_result, p_value




if __name__ == "__main__":
    df = make_dummy_data()
    result = compare_good_bad_by_step(df)
    print(result)
