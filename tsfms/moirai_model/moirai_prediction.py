import logging
import os
import sys

import numpy as np
import torch
from einops import rearrange


def make_prediction(context_df, ground_truth_df, df, args):
    """
    Moirai-specific prediction logic.
    We take the last `args.context_length` standardized prices (context_df)
    to predict the next `args.prediction_length` values.
    """
    # Convert the input data
    context = context_df[args.input_column].to_numpy()

    # If GPU is selected but no CUDA available, fallback to CPU
    backend = args.backend
    if backend == "gpu" and not torch.cuda.is_available():
        backend = "cpu"

    # Instantiate the Moirai model
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-base"),
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        patch_size=32,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    past_target = rearrange(torch.as_tensor(context, dtype=torch.float32), "t -> 1 t 1")
    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

    # Forecast
    forecast = model(
        past_target=past_target,
        past_observed_target=past_observed_target,
        past_is_pad=past_is_pad,
    )

    # Handle return_type
    if args.return_type == "median":
        # forecast[0] => shape: (num_samples, prediction_length)
        standardized_result = np.round(np.median(forecast[0], axis=0), decimals=4)
        result = destandardize_predictions(standardized_result, df)
    elif args.return_type == "median_quartile":
        median = np.median(forecast[0], axis=0)
        q1 = np.percentile(forecast[0], 25, axis=0)
        q3 = np.percentile(forecast[0], 75, axis=0)
        result = {
            "median": np.round(destandardize_predictions(median, df), decimals=4),
            "q1": np.round(destandardize_predictions(q1, df), decimals=4),
            "q3": np.round(destandardize_predictions(q3, df), decimals=4),
        }
    else:
        raise ValueError(f"Unknown return_type: {args.return_type}")

    return result


def destandardize_predictions(standardized_predictions, df):
    """
    Transform standardized predictions back to the original price scale,
    using the stats from `df["Close_denoised"]`.
    """
    return (
        standardized_predictions * df["Close_denoised"].std()
        + df["Close_denoised"].mean()
    )
