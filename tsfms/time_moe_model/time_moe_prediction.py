import numpy as np
import torch
from transformers import AutoModelForCausalLM


def make_prediction(context_df, ground_truth_df, df, args):
    """
    Time-MoE-specific prediction logic.
    We take the last `args.context_length` standardized prices (context_df)
    to predict the next `args.prediction_length` values.
    """

    context = context_df[args.input_column].to_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate the Time-MoE model
    model_timemoe = AutoModelForCausalLM.from_pretrained(
        "Maple728/TimeMoE-200M",
        device_map=device,
        trust_remote_code=True,
    )

    # Prepare context
    context_tensor = torch.tensor(
        context, dtype=torch.float32, device=device
    ).unsqueeze(0)

    # Generate forecast
    forecast = model_timemoe.generate(
        context_tensor, max_new_tokens=args.prediction_length
    )
    forecast_np = (
        forecast[:, -args.prediction_length :]
        .cpu()
        .numpy()
        .reshape(args.prediction_length, -1)
    )

    if args.return_type == "median":
        # The script suggests you were taking the median over forecast_np[:, 0]
        # (which is effectively a single column, but you can adapt if needed)
        result = np.round(np.median(forecast_np[:, 0], axis=0), decimals=4)
        # result = destandardize_predictions(result, df)
    elif args.return_type == "median_quartile":
        median = np.median(forecast_np[:, 0], axis=0)
        q1 = np.percentile(forecast_np[:, 0], 25, axis=0)
        q3 = np.percentile(forecast_np[:, 0], 75, axis=0)
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
