import argparse
import pandas as pd
import torch
import numpy as np
import logging
import os
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from einops import rearrange

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
    if args.utc:
        df.index = pd.to_datetime(df.index, utc=True)
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
            logging.info(f"Determined frequency: {frequency}")
        else:
            logging.info("Determining frequency of the dataframe...")
            frequency = determine_frequency(df)
            logging.info(f"Determined frequency: {frequency}")

        # Extract input and label for prediction
        if len(df) < args.context_length:
            raise ValueError(f"The provided dataset is too short for the specified context and prediction length. Required: {args.context_length}, Available: {len(df)}")
        
        inp = {
            "target": df[args.column].to_numpy()[-args.context_length:],  # Context length
            "start": df.index[0].to_period(freq=frequency),
        }

        # Initialize the model
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-{args.model_version}-{args.model_size}"),
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            patch_size=args.patch_size,
            num_samples=args.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        logging.info("Model initialized successfully.")

        # Prepare the input tensors
        past_target = rearrange(torch.as_tensor(inp["target"], dtype=torch.float32), "t -> 1 t 1")
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

        # Make the forecast
        forecast = model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        logging.info("Forecast generated successfully.")

        # Process the output based on the user's request
        if args.return_type == "median":
            result = np.round(np.median(forecast[0], axis=0), decimals=4)
        elif args.return_type == "median_quartile":
            median = np.median(forecast[0], axis=0)
            q1 = np.percentile(forecast[0], 25, axis=0)
            q3 = np.percentile(forecast[0], 75, axis=0)
            result = {
                "median": np.round(median, decimals=4),
                "q1": np.round(q1, decimals=4),
                "q3": np.round(q3, decimals=4),
            }
        else:
            raise ValueError(f"Unknown return_type: {args.return_type}")
        
        # Save the results in a new DataFrame
        last_index = df.index[-1]
        forecast_index = pd.date_range(start=last_index + pd.Timedelta(1, unit=frequency), periods=args.prediction_length, freq=frequency)

        if args.return_type == "median":
            result_df = pd.DataFrame(result, index=forecast_index, columns=["prediction"])
        elif args.return_type == "median_quartile":
            result_df = pd.DataFrame({
                "median": result["median"],
                "q1": result["q1"],
                "q3": result["q3"]
            }, index=forecast_index)

        # Save to CSV
        output_path = args.output if args.output else "forecast_results.csv"
        result_df.to_csv(output_path, index=True)
        logging.info(f"Prediction results saved to {output_path}")

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Moirai Forecast on a given time series dataset.")
    parser.add_argument("--path", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--utc", type=bool, required=True, help="Whether the dataframe is timezone-aware.")
    parser.add_argument("--column", type=str, required=True, help="Name of the column to be predicted.")
    parser.add_argument("--model_version", type=str, default="1.1-R", help="Model version to be used (e.g., 1.1-R).")
    parser.add_argument("--model_size", type=str, choices=["small", "base", "large"], default="small", help="Size of the model (small, base, large).")
    parser.add_argument("--prediction_length", type=int, default=48, help="Length of the prediction.")
    parser.add_argument("--context_length", type=int, default=168, help="Length of the context.")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples.")
    parser.add_argument("--return_type", type=str, choices=["median", "median_quartile"], default="median", help="What to return: median or median + quartile.")
    parser.add_argument("--output", type=str, help="Path to save the output CSV file.")
    parser.add_argument("--frequency", type=str, help="Frequency of the time series data (e.g., 'H' for hourly). If not provided, it will be inferred.")
    args = parser.parse_args()
    main(args)