# chronos_prediction.py
import logging

import numpy as np
import torch
from chronos import BaseChronosPipeline


def make_prediction(context_df, ground_truth_df, df, args):
    """
    Chronos-specific prediction logic.
    We take the last `args.context_length` standardized prices (context_df)
    to predict the next `args.prediction_length` values.
    """

    context = context_df[args.input_column].to_numpy()

    backend = args.backend
    if backend == "gpu" and not torch.cuda.is_available():
        backend = "cpu"

    model = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map=backend,
        torch_dtype=torch.bfloat16,
    )

    # Chronos expects a 1D tensor for the context
    context_tensor = torch.as_tensor(context, dtype=torch.float32)

    # Make forecast
    forecast = model.predict(context_tensor, args.prediction_length)
    # forecast has shape (batch_size=1, num_quantiles, prediction_length)
    # for example: [ (batch size is 1)
    #                  [-1.21, -1.18, -1.17, -1.13, -1.24, -1.21, -1.18, -1.17, -1.13, -1.24, -1.18, -1.17], # we have 12 values (12 horizons for every quantile)
    #                  [...], [...], [...], [...], [...], [...], [...], [...]
    #               ]

    forecast_np = forecast[0].numpy()  # shape: (num_quantiles, prediction_length)

    if args.return_type == "median":
        standardized_result = np.round(np.median(forecast_np, axis=0), decimals=4)
        result = destandardize_predictions(standardized_result, df)
    elif args.return_type == "median_quartile":
        median = np.median(forecast_np, axis=0)
        q1 = np.percentile(forecast_np, 25, axis=0)
        q3 = np.percentile(forecast_np, 75, axis=0)
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
