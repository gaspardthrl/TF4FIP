import numpy as np
import torch
from transformers import AutoModelForCausalLM

# python -m pipeline --path "../data/data_2024_2025_processed.csv" --input_column "Close" --output_column "Close" --prediction_length 12 --context_length 384 --frequency "H" --utc True --output "first_test.csv" --model_name "time_moe"


def make_prediction(
    context_list,
    model,
    args,
):
    """
    Generate forecasts for a batch of context windows using TimeMoE.
    context_list: list of DataFrames, each representing one context window
    model: the loaded TimeMoE model (AutoModelForCausalLM)
    args: script arguments (we need input_column, prediction_length, return_type, etc.)
    """

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.backend == "gpu") else "cpu"
    )

    # Prepare a list to store forecast outputs
    all_forecasts = []

    # For each window: standardize, keep track of mean and std for de-standardizing
    standardized_inputs = []
    means_stds = []
    for context_df in context_list:
        context = context_df[args.input_column].to_numpy()
        mean_val, std_val = context.mean(), context.std()
        # Handle edge case if std_val is 0 to avoid NaN
        if std_val == 0:
            std_val = 1e-8
        standardized = (context - mean_val) / std_val
        standardized_inputs.append(torch.tensor(standardized, dtype=torch.float32))
        means_stds.append((mean_val, std_val))

    # Stack into a single batch tensor
    # shape: (batch_size, context_length)
    batch_tensor = torch.stack(standardized_inputs).to(device)

    # Run generation for the entire batch
    with torch.no_grad():
        forecast_standardized = model.generate(
            batch_tensor, max_new_tokens=args.prediction_length
        )
        # forecast_standardized: shape is (batch_size, context_length + prediction_length)
        # Because model outputs the entire sequence. We'll pick off the last "prediction_length" tokens.

    # De-standardize each forecast in the batch
    # We iterate along the batch dimension of forecast_standardized
    for i, forecast_tensor in enumerate(forecast_standardized):
        mean_val, std_val = means_stds[i]
        # Convert to numpy
        forecast_np = forecast_tensor.detach().cpu().numpy()

        # The model output length is (context_length + prediction_length)
        # We only want the last `prediction_length` points
        forecast_np = forecast_np[-args.prediction_length :]

        # De-standardize
        forecast_np = forecast_np * std_val + mean_val

        # Now handle the 'return_type'
        if args.return_type == "median":
            final_result = forecast_np
        elif args.return_type == "median_quartile":
            median = np.median(forecast_np, axis=0)
            q1 = np.percentile(forecast_np, 25, axis=0)
            q3 = np.percentile(forecast_np, 75, axis=0)
            final_result = {
                "median": np.round(median, decimals=4),
                "q1": np.round(q1, decimals=4),
                "q3": np.round(q3, decimals=4),
            }
        else:
            raise ValueError(f"Unknown return_type: {args.return_type}")

        all_forecasts.append(final_result)

    return all_forecasts
