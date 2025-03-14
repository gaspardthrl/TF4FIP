import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from chronos_model.chronos_prediction import make_prediction as chronos_pred
from moirai_model.moirai_prediction import make_prediction as moirai_pred
from time_moe_model.time_moe_prediction import make_prediction as timemoe_pred
from transformers import AutoModelForCausalLM

# python -m pipeline --path "../data/data_2024_2025.csv" --input_column "Close_denoised" --output_column "Close" --prediction_length 3 --context_length 384 --frequency "H" --utc True --output "data_2024_2025_chronos.csv" --model_name "chronos"

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


def process_sliding_windows(df, args, predict_func, model, batch_size=64):
    """
    Run sliding-window predictions in batches of 64.
    """
    windows = list(
        sliding_window(df, args.context_length, args.prediction_length)
    )  # list of tuples: (context_df, ground_truth_df, idx)
    total_windows = len(windows)
    logging.info(f"Total windows: {total_windows}")

    results = []
    # We'll collect DataFrames for each batch
    for i in range(0, total_windows, batch_size):
        batch_slice = windows[i : i + batch_size]
        context_batch = [item[0] for item in batch_slice]  # list of context DFs
        ground_truth_batch = [
            item[1] for item in batch_slice
        ]  # list of ground-truth DFs
        idx_batch = [item[2] for item in batch_slice]  # list of indexes for logging

        # Get batch predictions using our batched function
        batch_predictions = predict_func(context_batch, model, args)

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

        # Save incremental backup every 100 or so windows
        if current_count % 100 == 0:
            with open("results_backup_incremental.pkl", "wb") as f:
                pickle.dump(results, f)
            logging.info(
                f"Incremental backup saved at {current_count} windows (batched)."
            )

    return results


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
        "time_moe": timemoe_pred,
    }

    if args.model_name not in model_map:
        raise ValueError(
            f"Unknown model_name: {args.model_name}. "
            "Valid choices are: moirai, chronos, time_moe."
        )

    predict_func = model_map[args.model_name]

    device = "cuda" if torch.cuda.is_available() and args.backend == "gpu" else "cpu"
    model = None

    # 4. We load the selected model once and then run sliding window predictions
    if args.model_name == "time_moe":
        model = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-200M",
            device_map=device,
            trust_remote_code=True,
        )
        model.to(device)
    elif args.model_name == "moirai":
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
    elif args.model_name == "chronos":
        model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    else:
        logging.error("Unknown model_name. Exiting...")
        return

    logging.info(f"Loading {args.model_name} model once for batched inference...")

    # Start predicition in batches
    logging.info("Starting sliding window predictions...")
    results = process_sliding_windows(
        df=df, args=args, model=model, predict_func=predict_func, batch_size=64
    )
    logging.info(
        f"Batched sliding window processing finished. {len(results)} windows found."
    )

    # 5. Build final result dataframe
    all_results = []
    for context_df, ground_truth_df, forecast, idx in results:
        # `forecast` here is either a NumPy array or a dict with median/quartiles
        print("forecast values: ", forecast)
        if isinstance(forecast, dict):
            logging.error("The median_quartile approach may contain errors.")
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
            )  # 1 if the prediction is in the same direction as the real price, -1 otherwise
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

    # Merge with original df
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
