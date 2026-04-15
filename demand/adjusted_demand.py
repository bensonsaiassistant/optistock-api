"""Adjusted demand forecasting — basic tier.

Takes raw historical sales data (as a pandas DataFrame) and produces
``final_avg`` and ``final_var`` — well-tuned moving average demand
estimates using rolling windows, outlier replacement, and demand
window selection based on sales frequency.

This module mirrors the original spreadsheet-based logic ported into
pandas/numba for performance.

Columns expected on input DataFrame:
    item, date, quantity, available, mercury_order_quantity

Columns added on output DataFrame:
    in_stock, item_group, replacement_value, adjusted_qty,
    60 avg/var/std/zsc/pct, 180 avg/var/std/zsc/pct,
    365 avg/var/std/zsc/pct, 730 avg/var/std/zsc/pct,
    60/180/365/730 adj_avg, 60/180/365/730 adj_var,
    non_zero, 60/180/365/730 non_zero_ct, final_avg, final_var
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NPY_EPSILON = 1e-8


def is_almost_integer(values: np.ndarray, tol: float = _NPY_EPSILON) -> bool:
    """Return True if every non-NaN value in *values* is within *tol* of an integer.

    Used to decide whether a column can be safely downcast to int dtype.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return True
    finite = np.isfinite(arr)
    if not np.any(finite):
        return True
    return bool(np.all(np.abs(arr[finite] - np.round(arr[finite])) <= tol))


def fill_dummy_data(
    df: pd.DataFrame, min_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Prepend ``min_date`` → first real date rows of zero-quantity, in-stock
    data for each item so rolling windows have context for new items.

    Parameters
    ----------
    df : DataFrame with columns ``item``, ``date`` (and others).
    min_date : earliest date to backfill.  Defaults to *max(date) - 365 days*.

    Returns
    -------
    DataFrame with dummy rows prepended per item, sorted by item/date.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()

    if min_date is None:
        min_date = max_date - pd.Timedelta(days=365)

    items = df["item"].unique()
    dummy_frames: list[pd.DataFrame] = []

    for item in items:
        item_rows = df[df["item"] == item]
        first_real = item_rows["date"].min()

        if first_real > min_date:
            # Build daily dummy rows from min_date up to (but not including) first_real
            dummy_dates = pd.date_range(min_date, first_real - pd.Timedelta(days=1), freq="D")
            dummy = pd.DataFrame(
                {
                    "item": item,
                    "date": dummy_dates,
                    "quantity": 0.0,
                    "available": 1.0,
                    "mercury_order_quantity": 0.0,
                }
            )
            dummy_frames.append(dummy)

    if dummy_frames:
        dummies = pd.concat(dummy_frames, ignore_index=True)
        df = pd.concat([dummies, df], ignore_index=True)

    df = df.sort_values(["item", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Numba-friendly rolling helpers
# ---------------------------------------------------------------------------

def _rolling_stat(
    group_key: pd.Series, values: pd.Series, window: int, stat: str
) -> np.ndarray:
    """Compute a rolling statistic grouped by *group_key* using numba.

    This wrapper handles the groupby-rolling boilerplate and returns
    an aligned numpy array (same length as input).
    """
    result = (
        values.groupby(group_key)
        .rolling(window=window, min_periods=0)
        .agg(stat, engine="numba", engine_kwargs={"parallel": True, "nopython": True})
        .reset_index(level=0, drop=True)
    )
    # result is a Series indexed by the original DataFrame position
    # but groupby-rolling may reorder, so we need to reindex
    return result.values


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def calculate_adjusted_demand(
    df: pd.DataFrame,
    min_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Enrich *df* with ``final_avg`` and ``final_var`` demand estimates.

    Logic (from original implementation):
    1. Backfill dummy zero-sales rows so new items aren't misleading.
    2. Compute in_stock, backorder flags.
    3. Rolling windows (14, 30, 60, 180, 365, 730 days): avg, var, std,
       z-score, pct.
    4. Outlier replacement: extreme z-scores (> per-window threshold) are
       replaced with a shifted 60-day average (clipped to ≥ 1).
    5. Adjusted averages: rolling means of ``adjusted_qty``.
    6. Demand window selection based on sales frequency (60/180/365/730).
    7. Backorder adjustment: increase window if backorders exist in that
       period.
    8. ``final_avg`` / ``final_var`` picked from the chosen window's
       adjusted stats, with variance floor (max of var, avg).
    9. NaN handling: out-of-stock periods set to NaN, then forward-filled
       within item groups, remaining NaNs filled with 0.
    10. Drop intermediate columns that aren't needed downstream.
    11. Filter out dummy rows (only keep date >= min_date).

    Parameters
    ----------
    df : DataFrame with columns ``item``, ``date``, ``quantity``,
         ``available``, ``mercury_order_quantity``.
    min_date : Earliest date to keep in output.  Defaults to *max(date) - 365
               days*.  Dummy rows before this are removed at the end.

    Returns
    -------
    Enriched DataFrame with ``final_avg`` and ``final_var`` columns.
    """
    # --- Z-score thresholds per window ---
    z_thresh = {14: 2, 30: 3.5, 60: 6.5, 180: 8, 365: 12, 730: 20}

    # --- Edge case: empty input ---
    if df.empty:
        return df.assign(
            final_avg=0.0,
            final_var=0.0,
        )

    # --- Step 0: Deep copy and fill dummy data ---
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Resolve min_date before filling dummy data
    max_date = df["date"].max()
    if min_date is None:
        min_date = max_date - pd.Timedelta(days=365)

    df = fill_dummy_data(df, min_date=min_date)

    # --- Step 1: In-stock and backorder flags ---
    df["in_stock"] = np.where(
        (df["available"] <= 0) & (df["quantity"] == 0), 0, 1
    )
    df["item_group"] = df["item"].astype(str) * df["in_stock"].astype(int)

    df["backorder_to_count"] = np.where(
        (df["available"] <= 0)
        & (df["quantity"] - df["mercury_order_quantity"] > 0),
        1,
        0,
    )

    df = df.sort_values(["item_group", "date"]).reset_index(drop=True)

    # --- Step 2: Rolling backorder counts ---
    for w in [30, 60, 180, 365, 730]:
        col = f"{w}_backorders"
        rolled = (
            df["backorder_to_count"]
            .groupby(df["item_group"])
            .rolling(window=w, min_periods=0)
            .sum(engine="numba", engine_kwargs={"parallel": True, "nopython": True})
        )
        # rolled has a MultiIndex (item_group, original_pos); we need values aligned
        df[col] = rolled.reset_index(level=0, drop=True).values

    # --- Step 3: Rolling stats per window ---
    windows = [14, 30, 60, 180, 365, 730]
    qty_col = df["quantity"]

    for w in windows:
        grp = qty_col.groupby(df["item_group"])
        rolling = grp.rolling(window=w, min_periods=0)

        df[f"{w} avg"] = rolling.mean(
            engine="numba", engine_kwargs={"parallel": True, "nopython": True}
        ).reset_index(level=0, drop=True).values
        df[f"{w} var"] = rolling.var(
            engine="numba", engine_kwargs={"parallel": True, "nopython": True}
        ).reset_index(level=0, drop=True).values
        df[f"{w} std"] = rolling.std(
            engine="numba", engine_kwargs={"parallel": True, "nopython": True}
        ).reset_index(level=0, drop=True).values
        # Z-score: (quantity - avg) / std
        df[f"{w} zsc"] = (qty_col - df[f"{w} avg"]) / df[f"{w} std"]
        # Pct: quantity / rolling sum
        roll_sum = rolling.sum(
            engine="numba", engine_kwargs={"parallel": True, "nopython": True}
        ).reset_index(level=0, drop=True).values
        df[f"{w} pct"] = qty_col / roll_sum

    # --- Step 4: Replacement value (shifted 60-day avg, clipped) ---
    df["replacement_value"] = (
        df.groupby("item_group")["60 avg"]
        .shift()
        .clip(lower=1)
        .round(0)
        .fillna(1)
    )

    # --- Step 5: Outlier-adjusted quantity ---
    df["adjusted_qty"] = df["quantity"].copy()

    for w in [60, 180, 365, 730]:
        thresh = z_thresh[w]
        # z-score outlier replacement
        df["adjusted_qty"] = np.where(
            df[f"{w} zsc"] > thresh, df["replacement_value"], df["adjusted_qty"]
        )
        # pct outlier replacement
        df["adjusted_qty"] = np.where(
            df[f"{w} pct"] > 0.5, df["replacement_value"], df["adjusted_qty"]
        )
        df["adjusted_qty"] = np.where(
            df[f"{w} pct"] > 0.9, 1, df["adjusted_qty"]
        )

    # --- Step 6: Adjusted rolling averages and variances ---
    adj_qty = df["adjusted_qty"]

    for w in [60, 180, 365, 730]:
        grp = adj_qty.groupby(df["item_group"])
        rolling = grp.rolling(window=w, min_periods=0)

        df[f"{w} adj_avg"] = rolling.mean(
            engine="numba", engine_kwargs={"parallel": True, "nopython": True}
        ).reset_index(level=0, drop=True).values
        df[f"{w} adj_var"] = rolling.var(
            engine="numba", engine_kwargs={"parallel": True, "nopython": True}
        ).reset_index(level=0, drop=True).values

    # --- Step 7: Non-zero count per window ---
    df["non_zero"] = np.where(df["adjusted_qty"] != 0, 1, 0)

    for w in [14, 30, 60, 180, 365, 730]:
        non_zero_count = (
            df["non_zero"]
            .groupby(df["item_group"])
            .rolling(window=w, min_periods=0)
            .sum(engine="numba", engine_kwargs={"parallel": True, "nopython": True})
        )
        df[f"{w} non_zero_ct"] = non_zero_count.reset_index(level=0, drop=True).values

    # --- Step 8: Demand window selection ---
    # Priority: 730 (least frequent) → 60 (most frequent)
    df["demand_window"] = np.where(
        (df["365 adj_avg"] < (2 / 365)) | (df["365 non_zero_ct"] < 2),
        4,  # 730-day window
        np.where(
            (df["180 adj_avg"] < (2 / 180)) | (df["180 non_zero_ct"] < 2),
            3,  # 365-day window
            np.where(
                (df["60 adj_avg"] < (2 / 60)) | (df["60 non_zero_ct"] < 2),
                2,  # 180-day window
                1,  # 60-day window
            ),
        ),
    )

    # --- Step 9: Backorder adjustment — increase window if backorders exist ---
    df["demand_window"] = np.where(
        (df["demand_window"] == 1)
        & (df["60_backorders"] > 0)
        & (df["60 non_zero_ct"] < 5),
        df["demand_window"] + 1,
        np.where(
            (df["demand_window"] == 2)
            & (df["180_backorders"] > 0)
            & (df["180 non_zero_ct"] < 5),
            df["demand_window"] + 1,
            np.where(
                (df["demand_window"] == 3)
                & (df["365_backorders"] > 0)
                & (df["365 non_zero_ct"] < 5),
                df["demand_window"] + 1,
                df["demand_window"],
            ),
        ),
    )

    # --- Step 10: Final avg/var from selected window ---
    df["final_avg"] = np.where(
        df["demand_window"] == 4,
        df["730 adj_avg"],
        np.where(
            df["demand_window"] == 3,
            df["365 adj_avg"],
            np.where(
                df["demand_window"] == 2,
                df["180 adj_avg"],
                df["60 adj_avg"],  # window 1 or fallback
            ),
        ),
    ).clip(min=0)

    df["final_var"] = np.where(
        df["demand_window"] == 4,
        df["730 adj_var"],
        np.where(
            df["demand_window"] == 3,
            df["365 adj_var"],
            np.where(
                df["demand_window"] == 2,
                df["180 adj_var"],
                df["60 adj_var"],  # window 1 or fallback
            ),
        ),
    ).clip(min=0)

    # --- Step 11: Variance floor ---
    df["final_var"] = np.where(
        df["final_var"] < df["final_avg"], df["final_avg"], df["final_var"]
    )

    # --- Step 12: Drop intermediate columns ---
    drop_cols = [
        "14 non_zero_ct",
        "30 non_zero_ct",
        "14 adj_var",
        "30 adj_var",
        "14 adj_avg",
        "30 adj_avg",
        "14 avg",
        "14 var",
        "14 std",
        "14 zsc",
        "14 pct",
        "30 avg",
        "30 var",
        "30 std",
        "30 zsc",
        "30 pct",
        "in_stock",
        "item_group",
        "backorder_to_count",
        "30_backorders",
        "60_backorders",
        "180_backorders",
        "365_backorders",
        "730_backorders",
        "demand_window",
    ]
    # Drop only columns that exist
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)

    # --- Step 13: Sort, filter dummy rows, handle NaNs ---
    df = df.sort_values(["item", "date"]).reset_index(drop=True)

    # Remove dummy rows (keep only date >= min_date)
    df = df[df["date"] >= min_date].reset_index(drop=True)

    columns_to_update = [
        "60 avg", "60 var", "60 std", "60 zsc", "60 pct",
        "180 avg", "180 var", "180 std", "180 zsc", "180 pct",
        "365 avg", "365 var", "365 std", "365 zsc", "365 pct",
        "730 avg", "730 var", "730 std", "730 zsc", "730 pct",
        "replacement_value", "adjusted_qty",
        "60 adj_avg", "180 adj_avg", "365 adj_avg", "730 adj_avg",
        "60 adj_var", "180 adj_var", "365 adj_var", "730 adj_var",
        "non_zero",
        "60 non_zero_ct", "180 non_zero_ct", "365 non_zero_ct", "730 non_zero_ct",
        "final_avg", "final_var",
    ]

    for col in columns_to_update:
        if col not in df.columns:
            continue
        # Set to NaN where in_stock would be 0 (OOS period).
        # Since we dropped in_stock, reconstruct it.
        oos_mask = (df["available"] <= 0) & (df["quantity"] == 0)
        df[col] = np.where(oos_mask, np.nan, df[col])
        # Forward-fill within item groups
        df[col] = df.groupby("item")[col].ffill()
        # Fill remaining NaNs with 0
        df[col] = df[col].fillna(0)
        # Downcast
        values = df[col].values
        if is_almost_integer(values):
            df[col] = pd.to_numeric(values, downcast="integer")
        else:
            df[col] = pd.to_numeric(values, downcast="float")

    return df


# ---------------------------------------------------------------------------
# Convenience API wrapper
# ---------------------------------------------------------------------------

def calculate_demand_from_history(
    historical_data: list[dict],
) -> tuple[float, float, str]:
    """High-level convenience function for the API layer.

    Takes a list of dicts with keys:
        ``date`` (ISO string), ``quantity``, ``available``, ``mercury_order_quantity``
    and returns ``(final_avg, final_var, source_label)`` for the last row
    of each unique item.

    If there's only one item, returns scalars for that item's latest demand
    estimate. If there are multiple items, aggregates by taking the mean of
    final_avg and final_var across items.

    Parameters
    ----------
    historical_data : list of dicts as described above.

    Returns
    -------
    tuple[float, float, str]
        (final_avg, final_var, "adjusted_demand")
    """
    if not historical_data:
        return 0.0, 0.0, "no_data"

    df = pd.DataFrame(historical_data)

    # Ensure required columns exist with defaults
    for col, default in [
        ("quantity", 0.0),
        ("available", 1.0),
        ("mercury_order_quantity", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Ensure item column exists
    if "item" not in df.columns:
        df["item"] = "DEFAULT"

    # Edge case: single row
    if len(df) == 1:
        qty = float(df["quantity"].iloc[0])
        if qty > 0:
            return qty, max(qty, 0.0), "adjusted_demand"
        return 0.0, 0.0, "no_sales"

    result = calculate_adjusted_demand(df)

    if result.empty:
        return 0.0, 0.0, "no_data"

    # Get last row per item, then average final_avg and final_var
    last_per_item = result.sort_values(["item", "date"]).groupby("item").last()
    final_avg = float(last_per_item["final_avg"].mean())
    final_var = float(last_per_item["final_var"].mean())

    return final_avg, final_var, "adjusted_demand"
