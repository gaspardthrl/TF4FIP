import argparse
import concurrent.futures
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch

from chronos_model.chronos_prediction import make_prediction as chronos_pred
from moirai_model.moirai_prediction import make_prediction as moirai_pred

# Single-window function from time_moe_model, used only if we don't do batch
# (You could remove this import if you do not plan to do single-window predictions for TimeMoE.)
from time_moe_model.time_moe_prediction import make_prediction as timemoe_pred

from transformers import AutoModelForCausalLM

###############################################################################
# Common utility functions
###############################################################################


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("forecast_app.log"), logging.StreamHandler()],
    )


def load_data(file_path, input_column, output_column):
    """Load the data from the CSV file and verify the required columns exist."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    required_columns = [input_column, output_column]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def determine_frequency(df, utc):
    """Infer frequency if not provided, optionally apply UTC time index."""
    if utc:
        df.index = pd.to_datetime(df.index, utc=True)
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        raise ValueError(
            "Could not determine the frequency of the dataframe index. "
            "Please specify the frequency explicitly."
        )
    return inferred_freq


def sliding_window(df, context_length, prediction_length):
    """
    Generator that yields (context_df, ground_truth_df, index_for_logging).
    context_df: last 'context_length' rows
    ground_truth_df: next 'prediction_length' rows
    """
    for start_idx in range(len(df) - context_length - prediction_length + 1):
        context_end = start_idx + context_length  # end of the context window
        pred_end = context_end + prediction_length  # end of the prediction window

        context_df = df.iloc[start_idx:context_end]
        ground_truth_df = df.iloc[context_end:pred_end]

        yield context_df, ground_truth_df, df.index[context_end]


###############################################################################
# BATCH PREDICTION FUNCTION for TIME_MOE
###############################################################################
# -------------------- BATCH CHANGE START --------------------

def batched_timemoe_prediction(
    context_list,
    model,
    args,
):
    """
    Generate forecasts for a *batch* of context windows using TimeMoE.
    context_list: list of DataFrames, each representing one context window
    model: the loaded TimeMoE model (AutoModelForCausalLM)
    args: script arguments (we need input_column, prediction_length, return_type, etc.)
    """

    device = torch.device("cuda" if (torch.cuda.is_available() and args.backend == "gpu") else "cpu")

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

    # Run generation *once* for the entire batch
    with torch.no_grad():
        forecast_standardized = model.generate(
            batch_tensor,
            max_new_tokens=args.prediction_length
        )
        # forecast_standardized: shape is (batch_size, context_length + prediction_length)
        # Because model outputs the entire sequence. We'll pick off the last "prediction_length" tokens.

    # De-standardize each forecast in the batch
    # We'll iterate along the batch dimension of forecast_standardized
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
            # Single-run generation is effectively just 'forecast_np'
            final_result = forecast_np
        elif args.return_type == "median_quartile":
            # If you truly need median/quartile from multiple runs,
            # you’d normally sample multiple times. Here, we’ll replicate:
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


def process_sliding_windows_batched(df, args, model, batch_size=64):
    """
    Run sliding-window predictions for TimeMoE in *batches* for efficiency.
    """
    windows = list(sliding_window(df, args.context_length, args.prediction_length))
    total_windows = len(windows)
    logging.info(f"Total windows: {total_windows}")

    results = []
    # We'll collect DataFrames for each batch
    for i in range(0, total_windows, batch_size):
        batch_slice = windows[i : i + batch_size]
        context_batch = [item[0] for item in batch_slice]  # list of context DFs
        ground_truth_batch = [item[1] for item in batch_slice]  # list of ground-truth DFs
        idx_batch = [item[2] for item in batch_slice]  # list of indexes for logging

        # Get batch predictions using our batched function
        batch_predictions = batched_timemoe_prediction(context_batch, model, args)

        # Combine predictions with the original (context_df, ground_truth_df, idx)
        for context_df, ground_truth_df, idx_val, forecast in zip(
            context_batch, ground_truth_batch, idx_batch, batch_predictions
        ):
            results.append((context_df, ground_truth_df, forecast, idx_val))

        # Log progress
        current_count = min(i + batch_size, total_windows)
        progress = (current_count / total_windows) * 100
        logging.info(
            f"BATCH {i // batch_size + 1} => Processed {current_count}/{total_windows} windows ({progress:.2f}%)."
        )

        # Optionally: Save incremental backup every 100 or so windows
        if current_count % 100 == 0:
            with open("results_backup_incremental.pkl", "wb") as f:
                pickle.dump(results, f)
            logging.info(
                f"Incremental backup saved at {current_count} windows (batched)."
            )

    return results

# -------------------- BATCH CHANGE END --------------------


###############################################################################
# SINGLE-WINDOW PREDICTION: for moirai & chronos (kept as is)
###############################################################################


def process_sliding_windows_single(df, args, prediction_func):
    """
    Original concurrency-based approach for single-window calls (moirai, chronos).
    """
    results = []
    count = 0  # Counter for processed windows
    total_windows = len(df) - args.context_length - args.prediction_length + 1

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures_dict = {
            executor.submit(prediction_func, context_df, ground_truth_df, df, args): (
                context_df,
                ground_truth_df,
                index,
            )
            for context_df, ground_truth_df, index in sliding_window(
                df, args.context_length, args.prediction_length
            )
        }

        remaining = set(futures_dict.keys())
        while remaining:
            done, remaining = concurrent.futures.wait(
                remaining, timeout=60, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                context_df, ground_truth_df, idx = futures_dict[future]
                try:
                    result = future.result()
                    results.append((context_df, ground_truth_df, result, idx))
                    count += 1
                    progress = (count / total_windows) * 100
                    logging.info(
                        f"Processed {count}/{total_windows} windows ({progress:.2f}%)."
                    )
                    if count % 100 == 0:
                        with open("results_backup_incremental.pkl", "wb") as f:
                            pickle.dump(results, f)
                        logging.info(f"Incremental backup saved at {count} windows.")
                except Exception as e:
                    logging.error(f"Error during prediction for window at {idx}: {e}")

    return results


###############################################################################
# MAIN
###############################################################################


def main(args):
    setup_logging()
    logging.info("Starting the forecasting process...")

    # 1. Load data
    df = load_data(args.path, args.input_column, args.output_column)
    logging.info("Data loaded successfully.")

    # 2. Handle frequency / UTC
    if args.frequency:
        freq = args.frequency
    else:
        freq = determine_frequency(df, args.utc)
    args.frequency = freq
    logging.info(f"Frequency: {args.frequency}")

    # 3. Decide which make_prediction function to use based on model_name
    model_map = {
        "moirai": moirai_pred,
        "chronos": chronos_pred,
        "time_moe": timemoe_pred,  # single-window function
    }

    if args.model_name not in model_map:
        raise ValueError(
            f"Unknown model_name: {args.model_name}. "
            "Valid choices are: moirai, chronos, time_moe."
        )

    # 4. If the user wants time_moe, we do batched approach; else single-window concurrency
    if args.model_name == "time_moe":
        # -------------------- BATCH CHANGE START --------------------
        logging.info("Loading TimeMoE model once for batched inference...")
        device = "cuda" if torch.cuda.is_available() and args.backend == "gpu" else "cpu"

        # Load the model ONCE
        model = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-200M",
            device_map=device,
            trust_remote_code=True,
        )
        model.to(device)

        # Use the batched approach
        logging.info("Starting sliding window predictions (batched TimeMoE)...")
        results = process_sliding_windows_batched(df, args, model, batch_size=64)
        logging.info(f"Batched sliding window processing finished. {len(results)} windows found.")
        # -------------------- BATCH CHANGE END --------------------
    else:
        # For moirai & chronos, keep the original concurrency approach
        prediction_func = model_map[args.model_name]
        logging.info(f"Starting sliding window predictions for {args.model_name} (single-window concurrent)...")
        results = process_sliding_windows_single(df, args, prediction_func)
        logging.info(f"Sliding Window processing finished. {len(results)} windows found.")

    # 5. Build final result dataframe
    all_results = []
    for context_df, ground_truth_df, forecast, idx in results:
        # `forecast` here is either a NumPy array or a dict with median/quartiles
        if isinstance(forecast, dict):
            # Means we used "median_quartile" approach
            # but your code below is written as if forecast is an array
            # For demonstration, let's handle that scenario carefully:
            if "median" in forecast:
                final_predictions = forecast["median"]
            else:
                raise ValueError("Unknown format for 'forecast' dictionary.")
        else:
            final_predictions = forecast

        # Compute APE, SIGN, etc.
        real_values = ground_truth_df[args.output_column].values

        APE = [
            100 * abs(x_real - x_pred) / (abs(x_real) if x_real != 0 else 1e-8)
            for x_real, x_pred in zip(real_values, final_predictions)
        ]

        SIGN = [
            (
                1 if np.sign(base_val - real) == np.sign(base_val - pred) else -1
            )
            for real, pred, base_val in zip(
                real_values,
                final_predictions,
                # last value of the context_df repeated for the length of the predictions
                [context_df[args.output_column].values[-1]] * len(final_predictions),
            )
        ]

        SCORE = [a * s for a, s in zip(APE, SIGN)]

        all_results.append(
            {
                "context_end": context_df.index[-1],
                "score": SCORE,
                "APE": APE,
                "SIGN": SIGN,
                "Result": final_predictions,
            }
        )

    # 6. Convert to a dataframe and write out
    results_df = pd.DataFrame(all_results)
    results_df.index = results_df["context_end"]
    results_df.sort_index(inplace=True)
    results_df.drop(["context_end"], axis=1, inplace=True)
    results_df.index.name = "Datetime"

    # Merge with original df (optional)
    final_df = pd.concat([df, results_df], axis=1)

    # Save to CSV
    output_path = (
        args.output
        or f"new_results_{args.model_name}_{args.path[:-4]}_{args.context_length}.csv"
    )
    final_df.to_csv(output_path, index=True)

    logging.info(f"Results saved to {output_path}")
    logging.info("Forecasting process completed.")


###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Forecast on a given time series dataset with multiple model backends."
    )
    parser.add_argument("--path", type=str, required=True, help="Path to CSV dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["moirai", "chronos", "time_moe"],
        help="Which model to run (moirai, chronos, or time_moe).",
    )

    # Common arguments used among all scripts
    parser.add_argument("--utc", type=bool, default=False, help="Timezone-aware data?")
    parser.add_argument(
        "--input_column",
        type=str,
        required=True,
        help="Name of the column to be predicted.",
    )
    parser.add_argument(
        "--output_column",
        type=str,
        required=True,
        help="Name of the column used for comparison.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device for inference (gpu or cpu).",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=3,
        help="Length of the horizon to forecast.",
    )
    parser.add_argument(
        "--context_length", type=int, default=384, help="History length (context)."
    )
    parser.add_argument(
        "--frequency",
        type=str,
        help="Frequency of the time series data (e.g. 'H'). If not provided, it will be inferred.",
    )
    parser.add_argument("--output", type=str, help="Path to save the output CSV file.")
    parser.add_argument(
        "--return_type",
        type=str,
        choices=["median", "median_quartile"],
        default="median",
        help="What to return: median or median + quartile.",
    )

    args = parser.parse_args()
    main(args)