import argparse
import concurrent.futures
import logging
import os

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from einops import rearrange
from tqdm import tqdm

# from uni2ts.src.uni2ts.model.moirai import MoiraiForecast, MoiraiModule
# from uni2ts.src.uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

# TODO
# 1. Modify the make_prediction to use Chronos instead of MoiraiForecast
## 1.1 Need to adapt the input and maybe the output it depends
# 2. Verify that we look for the median


# nohup python rolling_window.py --path "../../data/ES=F.csv" --column "Close_denoised_standardized" --prediction_length 16 --context_length 384 --frequency "H" --utc True


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("forecast_app.log"), logging.StreamHandler()],
    )


def load_data(file_path, target_column):
    # Load the data from the CSV file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found.")
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    if target_column not in df.columns:
        raise ValueError(
            f"The specified column '{target_column}' does not exist in the provided dataset."
        )
    return df


def determine_frequency(df):
    if args.utc:
        df.index = pd.to_datetime(df.index, utc=True)
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        raise ValueError(
            "Could not determine the frequency of the dataframe index. Please specify the frequency explicitly."
        )
    return inferred_freq


# Sliding window function
def sliding_window(df, context_length, prediction_length):
    """Generates context and prediction dataframes for each sliding window."""
    for start in range(len(df) - context_length - prediction_length + 1):
        context_end = start + context_length
        prediction_end = context_end + prediction_length

        context_df = df.iloc[start:context_end]
        prediction_df = df.iloc[context_end:prediction_end]

        yield context_df, prediction_df, df.index[context_end]


# Prediction wrapper function
def make_prediction(context_df, prediction_df, index, args):
    """Wrapper to call the prediction function for each sliding window."""

    inp = {
        "target": context_df[args.column].to_numpy(),
        "start": context_df.index[0].to_period(freq=args.frequency),
    }

    # Determine backend
    backend = args.backend
    if backend == "gpu" and not torch.cuda.is_available():
        logging.warning(
            "GPU backend specified but CUDA is not available. Falling back to CPU."
        )
        backend = "cpu"

    model = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large", device_map=backend, torch_dtype=torch.bfloat16
    )

    # Chronos expects a 1D tensor (or list of 1D tensors) for the context
    context_tensor = torch.as_tensor(inp["target"], dtype=torch.float32)

    forecast = model.predict(
        context_tensor,
        args.prediction_length,
        num_samples=args.num_samples,
    )

    forecast_np = forecast[0].numpy()
    if args.return_type == "median":
        result = np.round(np.median(forecast_np, axis=0), decimals=4)
    elif args.return_type == "median_quartile":
        median = np.median(forecast_np, axis=0)
        q1 = np.percentile(forecast_np, 25, axis=0)
        q3 = np.percentile(forecast_np, 75, axis=0)
        result = {
            "median": np.round(median, decimals=4),
            "q1": np.round(q1, decimals=4),
            "q3": np.round(q3, decimals=4),
        }
    else:
        raise ValueError(f"Unknown return_type: {args.return_type}")

    return result


# Concurrent processing of sliding windows
def process_sliding_windows(df, args):
    """Processes sliding windows concurrently using a ProcessPoolExecutor."""
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_window = {
            executor.submit(make_prediction, context_df, prediction_df, index, args): (
                context_df,
                prediction_df,
            )
            for context_df, prediction_df, index in sliding_window(
                df, args.context_length, args.prediction_length
            )
        }

        for future in concurrent.futures.as_completed(future_to_window):
            context_df, prediction_df = future_to_window[future]
            try:
                result = future.result()
                results.append((context_df, prediction_df, result))
            except Exception as e:
                logging.error(f"Error during prediction: {e}")

    return results


# Integrating the new functionality into main
def main(args):
    try:
        setup_logging()
        logging.info("Starting the forecasting process...")

        # Load the data
        df = load_data(args.path, args.column)
        logging.info("Data loaded successfully.")

        # Determine the frequency
        if args.frequency:
            frequency = args.frequency
            logging.info(f"Determined frequency: {frequency}")
        else:
            logging.info("Determining frequency of the dataframe...")
            frequency = determine_frequency(df)
            logging.info(f"Determined frequency: {frequency}")

        args.frequency = frequency  # Ensure frequency is set for downstream processes

        # For testing purposes, process only a subset of the data.
        # Ensure that we have enough rows: context_length + prediction_length + a few extra rows for sliding windows.
        subset_rows = args.context_length + args.prediction_length + 10
        df = df.head(subset_rows)
        logging.info(f"Processing a subset of the data: first {subset_rows} rows.")

        # Process sliding windows concurrently
        results = process_sliding_windows(df, args)
        logging.info(f"Sliding Window processed.")

        # Save results to a CSV file
        output_path = (
            args.output
            if args.output
            else f"new_results_{args.path[:-4]}_{args.context_length}.csv"
        )
        all_results = []
        for context_df, prediction_df, result in results:
            APE = list(
                map(
                    lambda x: 100 * abs(x[0] - x[1]) / abs(x[0]),
                    zip(prediction_df[args.column].values, result),
                )
            )

            SIGN = list(
                map(
                    lambda x: 1 if np.sign(x[2] - x[0]) == np.sign(x[2] - x[1]) else -1,
                    zip(
                        prediction_df[args.column].values,
                        result,
                        [context_df[args.column].values[-1]] * len(result),
                    ),
                )
            )

            MATRIX = list(
                map(
                    lambda x: (
                        [1, 0, 0, 0]
                        if np.sign(x[2] - x[0]) == np.sign(x[2] - x[1])
                        and np.sign(x[2] - x[0]) != -1
                        else (
                            [0, 1, 0, 0]
                            if np.sign(x[2] - x[0]) == np.sign(x[2] - x[1])
                            and np.sign(x[2] - x[0]) == -1
                            else (
                                [0, 0, 1, 0]
                                if np.sign(x[2] - x[0]) != np.sign(x[2] - x[1])
                                and np.sign(x[2] - x[0]) == -1
                                else (
                                    [0, 0, 0, 1]
                                    if np.sign(x[2] - x[0]) != np.sign(x[2] - x[1])
                                    and np.sign(x[2] - x[0]) != -1
                                    else [0, 0, 0, 0]
                                )
                            )
                        )
                    ),
                    zip(
                        prediction_df[args.column].values,
                        result,
                        [context_df[args.column].values[-1]] * len(result),
                    ),
                )
            )
            SCORE = list(map(lambda x: x[0] * x[1], zip(APE, SIGN)))

            all_results.append(
                {
                    "context_end": context_df.index[-1],
                    "score": SCORE,
                    "APE": APE,
                    "SIGN": SIGN,
                    "Result": result,
                    "MATRIX_1": MATRIX[0],
                    "MATRIX_2": MATRIX[1],
                    "MATRIX_3": MATRIX[2],
                    "MATRIX_4": MATRIX[3],
                    "MATRIX_5": MATRIX[4],
                    "MATRIX_6": MATRIX[5],
                    "MATRIX_7": MATRIX[6],
                    "MATRIX_8": MATRIX[7],
                    "MATRIX_9": MATRIX[8],
                    "MATRIX_10": MATRIX[9],
                    "MATRIX_11": MATRIX[10],
                    "MATRIX_12": MATRIX[11],
                }
            )

        logging.info("Processing finished")
        results_df = pd.DataFrame(all_results)
        results_df.index = results_df["context_end"]
        results_df.sort_index(inplace=True)
        results_df.drop(["context_end"], axis=1, inplace=True)
        results_df.index.name = "Datetime"
        pd.concat([df, results_df], axis=1).to_csv(output_path, index=True)
        logging.info(f"All sliding window results saved to {output_path}")

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Chronos Forecast on a given time series dataset."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the CSV file containing the dataset.",
    )
    parser.add_argument(
        "--utc",
        type=bool,
        required=True,
        help="Whether the dataframe is timezone-aware.",
    )
    parser.add_argument(
        "--column", type=str, required=True, help="Name of the column to be predicted."
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Backend to use for model inference (gpu or cpu).",
    )
    parser.add_argument(
        "--prediction_length", type=int, default=48, help="Length of the prediction."
    )
    parser.add_argument(
        "--context_length", type=int, default=168, help="Length of the context."
    )
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size.")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples."
    )
    parser.add_argument(
        "--return_type",
        type=str,
        choices=["median", "median_quartile"],
        default="median",
        help="What to return: median or median + quartile.",
    )
    parser.add_argument("--output", type=str, help="Path to save the output CSV file.")
    parser.add_argument(
        "--frequency",
        type=str,
        help="Frequency of the time series data (e.g., 'H' for hourly). If not provided, it will be inferred.",
    )
    args = parser.parse_args()
    main(args)
