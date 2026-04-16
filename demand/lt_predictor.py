"""ML-based lead time prediction for elite tier.

Uses ml-regression to predict future lead times from historical observations,
falling back to simple statistics when ML is unavailable or data is sparse.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from statistics import mean
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum data points before attempting ML lead-time prediction
MIN_LT_DATA_POINTS = 30

# Deviation guardrails (same pattern as demand ML predictor)
MAX_LT_DEVIATION_MULT = 3.0  # ML lt must be within 3x of historical mean


def _ensure_ml_regression_path() -> None:
    """Add ml-regression project to sys.path if not already present."""
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..', 'ml-regression')
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _build_lead_time_dataframe(
    historical_lead_times: list[float],
    item_id: str,
) -> "pl.DataFrame":
    """Build a minimal polars DataFrame for ml-regression lead time prediction.

    Creates synthetic time-indexed data where the target is the lead time value.
    The model learns temporal patterns to forecast future lead times.
    """
    import polars as pl
    import pandas as pd

    n = len(historical_lead_times)
    # Create synthetic dates — daily observations over n days
    dates = pd.date_range(end='2024-01-01', periods=n, freq='D')

    df = pd.DataFrame({
        'item': item_id,
        'date': dates,
        'lead_time': historical_lead_times,
    })
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days.astype(int)
    df['training_weights'] = 1.0

    # Date-derived features (same pattern as demand predictor)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter

    return pl.from_pandas(df)


def predict_lead_time_ml(
    historical_lead_times: list[float],
    item_id: str = "ITEM",
    min_data_points: int = MIN_LT_DATA_POINTS,
) -> tuple[float, float, str]:
    """Predict lead time using ml-regression (XGBoost time-series model).

    Args:
        historical_lead_times: Past lead time observations (in days).
        item_id: Item identifier for grouping.
        min_data_points: Minimum data points needed for ML prediction.

    Returns:
        (lead_time_estimate, lead_time_variance, source_label)
        source_label is one of:
        - "ml_regression"  — ML model ran successfully and passed guardrails
        - "historical_mean" — fell back to mean of historical lead times
        - "default"         — no historical data, used default of 14 days
    """
    # --- No data at all — use default ---
    if not historical_lead_times:
        return 14.0, 1.3, "default"

    historical_mean = mean(historical_lead_times)

    # --- Not enough data for ML — fall back to historical mean ---
    if len(historical_lead_times) < min_data_points:
        historical_var = float(np.var(historical_lead_times)) if len(historical_lead_times) > 1 else historical_mean
        historical_var = max(historical_var, historical_mean)  # variance floor
        return historical_mean, historical_var, "historical_mean"

    # --- Try ML regression ---
    try:
        _ensure_ml_regression_path()
        from ml_regression_polars import time_series_prediction_polars

        df_pl = _build_lead_time_dataframe(historical_lead_times, item_id)

        if df_pl.shape[0] < min_data_points:
            raise ValueError(f"Not enough rows after preprocessing: {df_pl.shape[0]}")

        # Define known/unknown columns for ml-regression
        known = []  # No known future covariates for lead time
        exclude_from_unknown = [
            'item', 'date', 'lead_time', 'training_weights',
            'time_idx', 'year', 'month', 'day', 'day_of_week',
            'day_of_year', 'week_of_year', 'quarter',
        ]
        unknown = [
            c for c in df_pl.columns
            if c not in exclude_from_unknown
        ]

        # Variance computation needs sufficient data
        compute_variance = df_pl.shape[0] >= 365

        with tempfile.TemporaryDirectory() as tmpdir:
            result_df, losses, models = time_series_prediction_polars(
                df=df_pl,
                target_variable='lead_time',
                known=known,
                unknown=unknown,
                steps_future=7,
                time_index_col='time_idx',
                group_col='item',
                steps_past=7,
                model_name='optistock_lead_time',
                model_type='xgboost',
                xgb_objective='reg:squarederror',
                use_optuna=False,
                compute_variance=compute_variance,
                roll_windows=[7, 30],
                simple_roll=True,
                compute_deltas=False,
                compute_accelerations=False,
                compute_interactions=False,
                shap_feature_pruning=False,
                use_gpu=False,
                verbose=False,
                use_all_data=True,
                output_dir=tmpdir,
            )

        # Extract predictions
        import pandas as pd
        prediction_col = 'lead_time_prediction'
        variance_col = 'variance'

        if prediction_col not in result_df.columns:
            pred_cols = [c for c in result_df.columns if 'prediction' in c.lower()]
            if pred_cols:
                prediction_col = pred_cols[0]
            else:
                raise ValueError(f"No prediction column found. Columns: {result_df.columns}")

        if isinstance(result_df, pd.DataFrame):
            last_pred = result_df.sort_values('time_idx').tail(7)
        else:
            import polars as pl
            last_pred = result_df.sort('time_idx').tail(7)

        if last_pred.shape[0] == 0:
            raise ValueError("No predictions returned")

        ml_lt = float(last_pred[prediction_col].mean())
        ml_lt = max(0.1, ml_lt)  # Can't have negative lead time

        # Variance
        if variance_col in result_df.columns:
            ml_var = float(last_pred[variance_col].mean())
            ml_var = max(0.1, ml_var)
        elif 'lead_time' in last_pred.columns:
            if isinstance(result_df, pd.DataFrame):
                residuals = (last_pred['lead_time'] - last_pred[prediction_col]) ** 2
            else:
                import polars as pl
                residuals = last_pred.select(
                    (pl.col('lead_time') - pl.col(prediction_col)) ** 2
                ).to_numpy().flatten()
            ml_var = float(np.mean(residuals)) if len(residuals) > 0 else ml_lt
            ml_var = max(0.1, ml_var)
        else:
            ml_var = ml_lt  # Use mean as variance floor

        # --- Deviation guardrail ---
        lt_ratio = ml_lt / historical_mean if historical_mean > 0 else float('inf')
        if lt_ratio > MAX_LT_DEVIATION_MULT or lt_ratio < (1 / MAX_LT_DEVIATION_MULT):
            logger.warning(
                f"ML lead time {ml_lt:.2f} deviates {lt_ratio:.1f}x from historical mean "
                f"{historical_mean:.2f} for item {item_id}. Falling back to historical_mean."
            )
            return historical_mean, max(ml_var, historical_mean), "historical_mean"

        # All guardrails passed
        logger.info(
            f"ML regression accepted for lead time item {item_id}: "
            f"lt={ml_lt:.2f} (historical_mean={historical_mean:.2f}, ratio={lt_ratio:.2f}), "
            f"var={ml_var:.2f}"
        )
        return ml_lt, ml_var, "ml_regression"

    except Exception as e:
        logger.warning(
            f"ML regression failed for lead time item {item_id}: {e}. "
            f"Falling back to historical_mean."
        )
        historical_var = float(np.var(historical_lead_times)) if len(historical_lead_times) > 1 else historical_mean
        historical_var = max(historical_var, historical_mean)
        return historical_mean, historical_var, "historical_mean"
