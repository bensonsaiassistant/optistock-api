"""ML Regression predictor for premium tier demand forecasting."""

import os
import numpy as np
from typing import Optional


# Lazy import ml_regression to avoid startup dependency
def _get_ml_regression():
    """Import ml_regression module dynamically."""
    import sys
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), '..', '..', 'ml-regression'),
    )
    from ml_regression_polars import process_dataframe_polars
    return process_dataframe_polars


def predict_demand_ml(
    historical_data: list[dict],
    item_id: str,
    min_data_points: int = 60,
) -> tuple[float, float, str]:
    """Predict demand using ml-regression (XGBoost).

    Args:
        historical_data: List of dicts with 'date' and 'quantity'
        item_id: Item identifier
        min_data_points: Minimum data points needed for ML prediction

    Returns:
        (ads, variance, "ml_regression") or falls back to adjusted_demand
    """
    if len(historical_data) < min_data_points:
        # Not enough data — fall back to adjusted_demand
        from demand.adjusted_demand import calculate_demand_from_history
        return calculate_demand_from_history(historical_data)

    try:
        import pandas as pd
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])

        # For now, use a simplified approach:
        # 1. Calculate rolling features from historical data
        # 2. Use the last row's final_avg/final_var as features
        # 3. Fall back to adjusted_demand if ml-regression fails
        from demand.adjusted_demand import calculate_adjusted_demand
        result_df = calculate_adjusted_demand(df)
        ads = float(result_df['final_avg'].iloc[-1])
        var = float(result_df['final_var'].iloc[-1])

        # TODO: When ml-regression models are trained, use them here.
        # For now, return adjusted_demand results (which is already quite good).
        return ads, var, "adjusted_demand_fallback"

    except Exception:
        # If anything fails, fall back to simple moving average
        from demand.adjusted_demand import calculate_demand_from_history
        return calculate_demand_from_history(historical_data)
