import logging
import os
import sys

import numpy as np
import torch
from einops import rearrange


def make_prediction(context_list, model, args):
    """
    Generate forecasts for a batch of context windows using Moirai.
    context_list: list of DataFrames, each representing one context window
    model: the loaded Moirai model (e.g., MoiraiForecast)
    args: script arguments (we need input_column, prediction_length, return_type, etc.)
    """

    # Decide on CPU/GPU device
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.backend == "gpu") else "cpu"
    )

    # Prepare arrays for each batch item
    all_past_targets = []
    all_observed_targets = []
    all_pads = []
    means_stds = (
        []
    )  # to keep track of mean & std for each context (for de-standardization)

    for context_df in context_list:
        # ---- Standardize this context window ----
        context = context_df[args.input_column].to_numpy()
        mean_val, std_val = context.mean(), context.std()

        # Avoid division by zero
        if std_val == 0:
            std_val = 1e-8

        # Standardize
        standardized = (context - mean_val) / std_val

        # Reshape to (context_length, 1) for Moirai’s (B, T, 1) input
        standardized = np.expand_dims(standardized, axis=-1)  # shape: (T, 1)

        # Convert to torch tensor
        standardized_tensor = torch.as_tensor(standardized, dtype=torch.float32)

        # Observed target (all True) and pad (all False)
        observed_tensor = torch.ones_like(standardized_tensor, dtype=torch.bool)
        pad_tensor = torch.zeros_like(standardized_tensor, dtype=torch.bool)

        # Append for batch stacking
        all_past_targets.append(standardized_tensor)
        all_observed_targets.append(observed_tensor)
        all_pads.append(pad_tensor)

        # Keep track for later de-standardization
        means_stds.append((mean_val, std_val))

    # ---- Stack everything into batched tensors ----
    # Each will have shape (batch_size, context_length, 1)
    batch_past_targets = torch.stack(all_past_targets, dim=0).to(device)
    batch_observed_targets = torch.stack(all_observed_targets, dim=0).to(device)
    batch_pads = torch.stack(all_pads, dim=0).to(device)

    # Squeeze the last dimension of the pad to match Moirai’s shape if needed
    batch_pads = batch_pads.squeeze(-1)  # shape => (batch_size, context_length)

    # ---- Run the Moirai model in one forward pass (batched) ----
    with torch.no_grad():
        # forecast is typically a tuple/list;
        # forecast[0] should have shape (batch_size, num_samples, prediction_length)
        forecast = model(
            past_target=batch_past_targets,
            past_observed_target=batch_observed_targets,
            past_is_pad=batch_pads,
        )

    # The first element in `forecast` is the array of generated samples
    # shape: (batch_size, num_samples, prediction_length)
    forecast_samples = forecast[0].cpu().numpy()

    # ---- De-standardize and post-process each forecast in the batch ----
    all_forecasts = []
    for i in range(forecast_samples.shape[0]):
        # forecast for i-th sample => shape (num_samples, prediction_length)
        item_forecast = forecast_samples[i]
        mean_val, std_val = means_stds[i]

        # De-standardize all samples: shape still (num_samples, prediction_length)
        item_forecast_destd = item_forecast * std_val + mean_val

        # Now handle the 'return_type'
        if args.return_type == "median":
            # median over the samples axis => shape (prediction_length,)
            final_result = np.median(item_forecast_destd, axis=0)
        elif args.return_type == "median_quartile":
            median = np.median(item_forecast_destd, axis=0)
            q1 = np.percentile(item_forecast_destd, 25, axis=0)
            q3 = np.percentile(item_forecast_destd, 75, axis=0)
            final_result = {
                "median": np.round(median, decimals=4),
                "q1": np.round(q1, decimals=4),
                "q3": np.round(q3, decimals=4),
            }
        else:
            raise ValueError(f"Unknown return_type: {args.return_type}")

        all_forecasts.append(final_result)

    return all_forecasts
