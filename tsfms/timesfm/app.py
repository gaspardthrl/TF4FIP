import argparse
import pandas as pd
import logging
import os
import timesfm
import torch

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("forecast_app.log"),
            logging.StreamHandler()
        ]
    )

def load_data(file_path, target_column):
    # Load the data from the CSV file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    if target_column not in df.columns:
        raise ValueError(f"The specified column '{target_column}' does not exist in the provided dataset.")
    return df

def determine_frequency(df):
    df.index = pd.to_datetime(df.index, utc=args.utc)
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        raise ValueError("Could not determine the frequency of the dataframe index. Please specify the frequency explicitly.")
    return inferred_freq

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
            logging.info(f"Frequency specified by user: {frequency}")
        else:
            logging.info("Determining frequency of the dataframe...")
            frequency = determine_frequency(df)
            logging.info(f"Determined frequency: {frequency}")

        # Validate input length for non-batch mode
        if len(df) < args.context_length:
            raise ValueError(f"The provided dataset is too short for the specified context and prediction length. Required: {args.context_length}, Available: {len(df)}")

        # Ensure context length is not greater than 512
        if args.context_length > 512:
            raise ValueError("The context length cannot be greater than 512.")

        # Determine backend
        backend = args.backend
        if backend == "gpu" and not torch.cuda.is_available():
            logging.warning("GPU backend specified but CUDA is not available. Falling back to CPU.")
            backend = "cpu"

        # Initialize TimesFm model
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=args.per_core_batch_size,
                horizon_len=args.prediction_length,
                context_len=args.context_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=args.huggingface_repo_id
            ),
        )
        logging.info("TimesFm model initialized successfully.")

        # Prepare the input DataFrame
        if args.batch:
            if not args.unique_id:
                raise ValueError("Batch mode requires specifying the unique_id column.")
            if args.unique_id not in df.columns:
                raise ValueError(f"The specified unique_id column '{args.unique_id}' does not exist in the provided dataset.")
            input_df = df[[args.column, args.unique_id]].copy()
            input_df.rename(columns={args.unique_id: "unique_id"}, inplace=True)
            input_df["ds"] = pd.to_datetime(df.index, utc=args.utc)
            logging.info("Input DataFrame prepared for batch forecasting.")
        else:
            input_df = df[[args.column]].copy()
            input_df["ds"] = pd.to_datetime(df.index, utc=args.utc)
            input_df["unique_id"] = [0] * len(input_df)
            logging.info("Input DataFrame prepared for single time series forecasting.")

        # Run the forecast
        forecast_result = tfm.forecast_on_df(
            inputs=input_df,
            freq=frequency,
            value_name=args.column,
            num_jobs=args.num_jobs,
        )
        logging.info("Forecast generated successfully.")

        # Save the results
        output_path = args.output if args.output else "forecast_results.csv"
        forecast_result.set_index(forecast_result["ds"], inplace=True)
        forecast_result.index.name = df.index.name

        if args.batch:
            forecast_result.rename(columns={"unique_id": args.unique_id}, inplace=True)
        else:
            forecast_result.drop("unique_id", axis=1, inplace=True)
        
        forecast_result.drop("ds", axis=1, inplace=True)
        forecast_result.to_csv(output_path, index=True)

        logging.info(f"Prediction results saved to {output_path}")

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TimesFm Forecast on a given time series dataset.")
    parser.add_argument("--path", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--utc", type=bool, required=True, help="Whether the dataframe is timezone-aware.")
    parser.add_argument("--column", type=str, required=True, help="Name of the column to be predicted.")
    parser.add_argument("--prediction_length", type=int, default=48, help="Length of the prediction.")
    parser.add_argument("--context_length", type=int, default=168, help="Length of the context. Must be <= 512.")
    parser.add_argument("--output", type=str, help="Path to save the output CSV file.")
    parser.add_argument("--frequency", type=str, help="Frequency of the time series data (e.g., 'H' for hourly). If not provided, it will be inferred.")
    parser.add_argument("--per_core_batch_size", type=int, default=32, help="Batch size per core for model training and inference.")
    parser.add_argument("--num_jobs", type=int, default=-1, help="Number of jobs to run in parallel for forecasting.")
    parser.add_argument("--huggingface_repo_id", type=str, default="google/timesfm-1.0-200m-pytorch", help="HuggingFace repository ID for the TimesFm model checkpoint.")
    parser.add_argument("--backend", type=str, choices=["gpu", "cpu"], default="gpu", help="Backend to use for model inference (gpu or cpu).")
    parser.add_argument("--batch", action="store_true", help="Indicates if the dataset contains multiple time series (batch mode).")
    parser.add_argument("--unique_id", type=str, help="Name of the column representing unique IDs for different time series in batch mode.")
    args = parser.parse_args()
    main(args)
