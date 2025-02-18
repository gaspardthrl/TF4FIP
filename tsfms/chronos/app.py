import argparse
import pandas as pd
import logging
import os
import torch
from sklearn.preprocessing import StandardScaler
from chronos import ChronosPipeline
import numpy as np

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

        # Validate input length
        if len(df) < args.context_length:
            raise ValueError(f"The provided dataset is too short for the specified context length. Required: {args.context_length}, Available: {len(df)}")

        # Determine backend
        backend = args.backend
        if backend == "gpu" and not torch.cuda.is_available():
            logging.warning("GPU backend specified but CUDA is not available. Falling back to CPU.")
            backend = "cpu"

        # Load ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            args.model,
            device_map=backend,
            torch_dtype=torch.bfloat16
        )
        logging.info("Chronos model initialized successfully.")

        # Prepare the input DataFrame
        if args.batch:
            raise ValueError("Batch mode is currently not supported in this Chronos-based implementation.")
        else:
            context_df = df.tail(args.context_length).copy()
            scaler = StandardScaler()
            context_df[args.column] = scaler.fit_transform(context_df[[args.column]])
            logging.info("Input DataFrame prepared for single time series forecasting.")

        # Prepare the context for prediction
        context = torch.tensor(context_df[args.column].values, dtype=torch.float32)

        # Run the forecast
        forecast = pipeline.predict(context, args.prediction_length)
        logging.info("Forecast generated successfully.")

        # Convert forecast from torch tensor to numpy array for visualization
        forecast_numpy = forecast[0].numpy()
        logging.info("Forecast converted successfully.")

        # Calculate quantiles for visualization (still in the scaled format)
        #TODO: Allow for additional quantile
        low, median, high = np.quantile(forecast_numpy, [0.1, 0.5, 0.9], axis=0)
        logging.info("Quantile generated successfully.")

        # Inverse scaling of the predictions
        median_inv = scaler.inverse_transform(median.reshape(-1, 1)).flatten()
        low_inv = scaler.inverse_transform(low.reshape(-1, 1)).flatten()
        high_inv = scaler.inverse_transform(high.reshape(-1, 1)).flatten()

        logging.info("Forecast invert-scaled successfully.")

        # Create forecast index for plotting or saving
        forecast_index = pd.date_range(start=context_df.index[-1] + pd.Timedelta(hours=1), periods=args.prediction_length, freq=frequency)
        logging.info("Forecast index updated successfully.")
        
        forecast_result = pd.DataFrame({
            "low": low_inv,
            "median": median_inv,
            "high": high_inv
        }, index=forecast_index)

        logging.info("Forecast converted to pandas dataframe successfully.")

        # Save the results
        output_path = args.output if args.output else "forecast_results.csv"
        forecast_result.to_csv(output_path, index=True)
        logging.info(f"Prediction results saved to {output_path}")

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chronos Forecast on a given time series dataset.")
    parser.add_argument("--path", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--utc", type=bool, required=True, help="Whether the dataframe is timezone-aware.")
    parser.add_argument("--column", type=str, required=True, help="Name of the column to be predicted.")
    parser.add_argument("--prediction_length", type=int, default=12, help="Length of the prediction.")
    parser.add_argument("--context_length", type=int, default=64, help="Length of the context. Must be <= length of dataset.")
    parser.add_argument("--output", type=str, help="Path to save the output CSV file.")
    parser.add_argument("--frequency", type=str, help="Frequency of the time series data (e.g., 'H' for hourly). If not provided, it will be inferred.")
    parser.add_argument("--backend", type=str, choices=["gpu", "cpu"], default="gpu", help="Backend to use for model inference (gpu or cpu).")
    parser.add_argument("--batch", action="store_true", help="Indicates if the dataset contains multiple time series (batch mode).")
    parser.add_argument("--model", type=str, default="amazon/chronos-bolt-small", help="Name of the Chronos model to be used.")
    args = parser.parse_args()
    main(args)
   