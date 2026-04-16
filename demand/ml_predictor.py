"""ML Regression predictor for premium tier demand forecasting.

Wraps ml-regression (XGBoost time-series model) to predict future demand
from historical data. Includes a safety guardrail: if ML predictions deviate
too far from the adjusted_demand (rolling average) baseline, falls back to
the baseline.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deviation guardrail thresholds
# ---------------------------------------------------------------------------

# If ML-predicted ads deviates from adjusted_demand ads by more than this
# multiplier, we reject the ML result and fall back.
MAX_ADS_DEVIATION_MULT = 3.0      # ML ads must be within 3x of baseline
MAX_VAR_DEVIATION_MULT = 5.0      # ML var must be within 5x of baseline

# Minimum historical data points before we even try ML
MIN_DATA_POINTS = 60


def _ensure_ml_regression_path() -> None:
    """Add ml-regression project to sys.path if not already present."""
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..', 'ml-regression')
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _build_minimal_dataframe(
    historical_data: list[dict],
    item_id: str,
) -> "pl.DataFrame":
    """Build a minimal polars DataFrame for ml-regression.

    Required columns:
    - item: item identifier (group_col)
    - date: datetime (time_index_col)
    - quantity: daily sales (target_variable)
    - time_idx: integer day index from the first date
    - training_weights: all 1.0
    """
    import polars as pl
    import pandas as pd

    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Ensure all required columns exist
    for col, default in [
        ('quantity', 0.0),
        ('available', 1.0),
        ('mercury_order_quantity', 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    df['item'] = item_id

    # time_idx = days since first date
    first_date = df['date'].min()
    df['time_idx'] = (df['date'] - first_date).dt.days.astype(int)

    df['training_weights'] = 1.0

    # Add date-derived features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter

    return pl.from_pandas(df)


def predict_demand_ml(
    historical_data: list[dict],
    item_id: str,
    min_data_points: int = MIN_DATA_POINTS,
) -> tuple[float, float, str]:
    """Predict demand using ml-regression (XGBoost time-series model).

    Args:
        historical_data: List of dicts with 'date', 'quantity', etc.
        item_id: Item identifier.
        min_data_points: Minimum data points needed for ML prediction.

    Returns:
        (ads, variance, source_label)
        source_label is one of:
        - "ml_regression"  — ML model ran successfully and passed guardrails
        - "adjusted_demand" — fell back to rolling average (insufficient data or deviation too large)
        - "insufficient_data" — not enough data for any model
    """
    # --- Not enough data — fall back immediately ---
    if len(historical_data) < min_data_points:
        from demand.adjusted_demand import calculate_demand_from_history
        ads, var, _ = calculate_demand_from_history(historical_data)
        return ads, var, "adjusted_demand"

    # --- Get baseline from adjusted_demand for comparison ---
    from demand.adjusted_demand import calculate_demand_from_history
    baseline_ads, baseline_var, baseline_source = calculate_demand_from_history(historical_data)

    if baseline_ads <= 0:
        # No demand signal at all — ML won't help
        return baseline_ads, baseline_var, baseline_source

    # --- Try ML regression ---
    try:
        _ensure_ml_regression_path()

        from ml_regression_polars import time_series_prediction_polars

        df_pl = _build_minimal_dataframe(historical_data, item_id)

        if df_pl.shape[0] < min_data_points:
            raise ValueError(f"Not enough rows after preprocessing: {df_pl.shape[0]}")

        # Define known/unknown columns for ml-regression
        known = ['sum_increased_traffic', 'sum_price_reduction']
        # Filter known to only columns that exist in df_pl
        known = [c for c in known if c in df_pl.columns]

        exclude_from_unknown = [
            'item', 'date', 'quantity', 'training_weights',
            'available', 'mercury_order_quantity',
            'time_idx', 'year', 'month', 'day', 'day_of_week',
            'day_of_year', 'week_of_year', 'quarter',
        ]
        unknown = [
            c for c in df_pl.columns
            if c not in exclude_from_unknown and c != 'quantity'
        ]

        # Variance computation needs sufficient data (fails with small datasets)
        compute_variance = df_pl.shape[0] >= 365  # At least 1 year of daily data

        # Use a temp dir for model output (don't pollute workspace)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run ML prediction (predict 7 days ahead — one week)
            result_df, losses, models = time_series_prediction_polars(
                df=df_pl,
                target_variable='quantity',
                known=known,
                unknown=unknown,
                steps_future=7,
                time_index_col='time_idx',
                group_col='item',
                steps_past=7,
                model_name='optistock_premium',
                model_type='xgboost',
                xgb_objective='reg:squarederror',
                use_optuna=False,           # Skip hyperparameter tuning for speed
                compute_variance=compute_variance,  # Only if enough data
                roll_windows=[7, 30],
                simple_roll=True,
                compute_deltas=False,
                compute_accelerations=False,
                compute_interactions=False,
                shap_feature_pruning=False,
                use_gpu=False,
                verbose=False,
                use_all_data=True,          # Train on all available data
                output_dir=tmpdir,
            )

        # Extract predictions from the result
        # result_df has columns like 'quantity_prediction', 'horizon', 'item', etc.
        prediction_col = 'quantity_prediction'
        variance_col = 'variance'

        if prediction_col not in result_df.columns:
            # Try alternative column names
            pred_cols = [c for c in result_df.columns if 'prediction' in c.lower()]
            if pred_cols:
                prediction_col = pred_cols[0]
            else:
                raise ValueError(f"No prediction column found in result. Columns: {result_df.columns}")

        # Get the predicted values (last row = latest prediction)
        # Note: ml-regression returns a pandas DataFrame (not polars)
        import pandas as pd
        if isinstance(result_df, pd.DataFrame):
            last_pred = result_df.sort_values('time_idx').tail(7)
        else:
            # Fallback for polars (shouldn't happen with current ml-regression)
            import polars as pl
            last_pred = result_df.sort('time_idx').tail(7)

        if last_pred.shape[0] == 0:
            raise ValueError("No predictions returned for item")

        ml_ads = float(last_pred[prediction_col].mean())
        ml_ads = max(0, ml_ads)  # Can't have negative demand

        if variance_col in result_df.columns:
            ml_var = float(last_pred[variance_col].mean())
            ml_var = max(0, ml_var)
        elif 'quantity' in result_df.columns:
            # Fallback: use variance from prediction residuals
            if isinstance(result_df, pd.DataFrame):
                residuals = (last_pred['quantity'] - last_pred[prediction_col]) ** 2
            else:
                import polars as pl
                residuals = last_pred.select(
                    (pl.col('quantity') - pl.col(prediction_col)) ** 2
                ).to_numpy().flatten()
            ml_var = float(np.mean(residuals)) if len(residuals) > 0 else ml_ads
            ml_var = max(0, ml_var)
        else:
            # No variance info available — use mean as variance floor
            ml_var = ml_ads

        # --- Variance floor ---
        if ml_var < ml_ads:
            ml_var = ml_ads

        # --- Deviation guardrail ---
        ads_ratio = ml_ads / baseline_ads if baseline_ads > 0 else float('inf')
        var_ratio = ml_var / baseline_var if baseline_var > 0 else float('inf')

        if ads_ratio > MAX_ADS_DEVIATION_MULT or ads_ratio < (1 / MAX_ADS_DEVIATION_MULT):
            logger.warning(
                f"ML ads {ml_ads:.2f} deviates {ads_ratio:.1f}x from baseline {baseline_ads:.2f} "
                f"for item {item_id}. Falling back to adjusted_demand."
            )
            return baseline_ads, baseline_var, "adjusted_demand"

        if var_ratio > MAX_VAR_DEVIATION_MULT or var_ratio < (1 / MAX_VAR_DEVIATION_MULT):
            logger.warning(
                f"ML var {ml_var:.2f} deviates {var_ratio:.1f}x from baseline {baseline_var:.2f} "
                f"for item {item_id}. Falling back to adjusted_demand."
            )
            return baseline_ads, baseline_var, "adjusted_demand"

        # All guardrails passed — use ML result
        logger.info(
            f"ML regression accepted for item {item_id}: "
            f"ads={ml_ads:.2f} (baseline={baseline_ads:.2f}, ratio={ads_ratio:.2f}), "
            f"var={ml_var:.2f} (baseline={baseline_var:.2f}, ratio={var_ratio:.2f})"
        )
        return ml_ads, ml_var, "ml_regression"

    except Exception as e:
        logger.warning(
            f"ML regression failed for item {item_id}: {e}. "
            f"Falling back to adjusted_demand."
        )
        return baseline_ads, baseline_var, "adjusted_demand"
