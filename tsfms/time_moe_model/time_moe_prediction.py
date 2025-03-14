import numpy as np
import torch
from transformers import AutoModelForCausalLM

# python -m pipeline --path "../data/data_2024_2025_processed.csv" --input_column "Close" --output_column "Close" --prediction_length 12 --context_length 384 --frequency "H" --utc True --output "first_test.csv" --model_name "time_moe"


def batched_make_prediction(context_list, args, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Standardize each context and create a batch tensor.
    standardized_contexts = []
    for context_df in context_list:
        context = context_df[args.input_column].to_numpy()
        # Standardize each context individually.
        standardized = (context - context.mean()) / context.std()
        standardized_contexts.append(torch.tensor(standardized, dtype=torch.float32))
    
    # Stack into a batch and move to the device.
    context_tensor = torch.stack(standardized_contexts).to(device)

    # Generate forecast for the whole batch.
    forecast_standardized = model.generate(
        context_tensor, max_new_tokens=args.prediction_length
    )

    # Post-process each forecast: de-standardize and keep only the forecast part.
    results = []
    for i, forecast_tensor in enumerate(forecast_standardized):
        # Note: The context for each sample needs its own mean and std.
        context = context_list[i][args.input_column].to_numpy()
        mean, std = context.mean(), context.std()
        forecast = forecast_tensor.cpu().detach().numpy() * std + mean
        forecast = forecast[-args.prediction_length:]

        if args.return_type == "median":
            result = forecast
        elif args.return_type == "median_quartile":
            median = np.median(forecast, axis=0)
            q1 = np.percentile(forecast, 25, axis=0)
            q3 = np.percentile(forecast, 75, axis=0)
            result = {
                "median": np.round(median, decimals=4),
                "q1": np.round(q1, decimals=4),
                "q3": np.round(q3, decimals=4),
            }
        else:
            raise ValueError(f"Unknown return_type: {args.return_type}")
        results.append(result)

    return results

# def make_prediction(context_df, ground_truth_df, df, args):
#     """
#     Time-MoE-specific prediction logic.
#     We take the last `args.context_length` standardized prices (context_df)
#     to predict the next `args.prediction_length` values.
#     """

#     # Set the device to GPU if available
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Time-MoE model requires standardized context
#     context = context_df[args.input_column].to_numpy()
#     context_standardized = (context - context.mean()) / context.std()
#     context_standardized_tensor = torch.tensor(
#         context_standardized, dtype=torch.float32, device=device
#     ).unsqueeze(0)

#     # Instantiate the Time-MoE model
#     model_timemoe = AutoModelForCausalLM.from_pretrained(
#         "Maple728/TimeMoE-200M",
#         device_map=device,
#         trust_remote_code=True,
#     )

#     # Generate forecast, the output is standardized
#     forecast_standardized = model_timemoe.generate(
#         context_standardized_tensor, max_new_tokens=args.prediction_length
#     )

#     # De-standardize the forecast
#     forecast = forecast_standardized * context.std() + context.mean()
#     forecast = (
#         forecast.squeeze().numpy()
#     )  # Remove the batch dimension and the tensor shape
#     forecast = forecast[
#         -args.prediction_length :
#     ]  # Keep only the last prediction_length values. Meaning, we remove the context_length values from the output

#     if args.return_type == "median":
#         result = forecast
#     elif args.return_type == "median_quartile":
#         median = np.median(forecast, axis=0)
#         q1 = np.percentile(forecast, 25, axis=0)
#         q3 = np.percentile(forecast, 75, axis=0)
#         result = {
#             "median": np.round(median, decimals=4),
#             "q1": np.round(q1, decimals=4),
#             "q3": np.round(q3, decimals=4),
#         }
#     else:
#         raise ValueError(f"Unknown return_type: {args.return_type}")

#     return result