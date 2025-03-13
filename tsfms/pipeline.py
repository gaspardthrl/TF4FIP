# main.py

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
from time_moe_model.time_moe_prediction import make_prediction as timemoe_pred

# python -m pipeline --path "data/data_2024_2025_processed.csv" --input_column "Close" --output_column "Close" --prediction_length 12 --context_length 384 --frequency "H" --utc True --output "first_test.csv" --model_name "time_moe"

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
        ground_truth_df = df.iloc[
            context_end:pred_end
        ]  # prediction_df represents the actual future values that the model is trying to predict

        yield context_df, ground_truth_df, df.index[context_end]


###############################################################################
# Main logic
###############################################################################


def process_sliding_windows(df, args, prediction_func):
    """
    Processes sliding windows concurrently.
    Calls the model-specific `prediction_func` for each window, collects results, and returns them.
    For an hour h, take the last context_lenght (384 by default) values corresponding to hours, to predict the next prediction_length (12 by default) values for hours h, h+1, ..., h+11.
    """
    results = []
    # You could tune max_workers if needed:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_window = {
            executor.submit(prediction_func, context_df, ground_truth_df, df, args): (
                context_df,
                ground_truth_df,
                index,
            )
            for context_df, ground_truth_df, index in sliding_window(
                df, args.context_length, args.prediction_length
            )
        }

        for future in concurrent.futures.as_completed(future_to_window):
            context_df, ground_truth_df, idx = future_to_window[future]
            try:
                result = future.result()
                results.append((context_df, ground_truth_df, result, idx))
            except Exception as e:
                logging.error(f"Error during prediction: {e}")

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
    prediction_func = model_map[args.model_name]

    # 4. Sliding window predictions (concurrent)
    results = process_sliding_windows(df, args, prediction_func)
    logging.info(f"Sliding Window processing finished. {len(results)} windows found.")

    # Save results for backup
    with open("results_backup.pkl", "wb") as f:
        pickle.dump(results, f)
    logging.info("Intermediate results saved successfully.")

    # 5. Build final result dataframe
    all_results = []

    print("\n Pipeline")

    for context_df, ground_truth_df, result, idx in results:
        # result is either a single array or a dict with median, q1, q3
        print("ground truth values")
        print(ground_truth_df[args.output_column].values)
        print(result)
        print("result")

        final_predictions = result

        # Compute APE, SIGN, etc.
        APE = [
            100 * abs(x_real - x_pred) / abs(x_real)
            for x_real, x_pred in zip(
                ground_truth_df[args.output_column].values, final_predictions
            )
        ]

        SIGN = [
            (
                1 if np.sign(base_val - real) == np.sign(base_val - pred) else -1
            )  # 1 if the prediction is in the same direction as the real price, -1 otherwise
            for real, pred, base_val in zip(
                ground_truth_df[args.output_column].values,
                final_predictions,
                [context_df[args.output_column].values[-1]]
                * len(
                    final_predictions
                ),  # last value of the context_df repeated for the length of the predictions
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
        default=3,  # 12 only required for bucket analyis
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
