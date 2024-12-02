import argparse
import pandas as pd
import torch
import numpy as np
import logging
import os
from transformers import AutoModelForCausalLM

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("forecast_timemoe_app.log"),
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
    else:
        df.index = pd.to_datetime(df.index, utc=False)
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        raise ValueError("Could not determine the frequency of the dataframe index. Please specify the frequency explicitly.")
    return inferred_freq

def main(args):
    try:
        setup_logging()
        logging.info("Starting the Time-MoE forecasting process...")

        # Load the data
        df = load_data(args.path, args.column)
        logging.info("Data loaded successfully.")

        # Determine the frequency
        if args.frequency:
            frequency = args.frequency
        else:
            logging.info("Determining frequency of the dataframe...")
            frequency = determine_frequency(df)
            logging.info(f"Determined frequency: {frequency}")

        # Extract input for prediction
        if len(df) < args.context_length:
            raise ValueError(f"The provided dataset is too short for the specified context length. Required: {args.context_length}, Available: {len(df)}")

        # Initialize Time-MoE model

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            logging.warning("CUDA is not available. Running on CPU, which may be slower.")
        else:
            logging.warning("CUDA available.")

        model_timemoe = AutoModelForCausalLM.from_pretrained(
            'Maple728/TimeMoE-200M',
            device_map=device,
            trust_remote_code=True,
        )
        logging.info("Time-MoE model initialized successfully.")
        
        # Prepare the Time-MoE input
        seqs = df[args.column].to_numpy()[-args.context_length:]
        seqs_tensor = torch.tensor(seqs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, context_length]
        mean, std = seqs_tensor.mean(dim=-1, keepdim=True), seqs_tensor.std(dim=-1, keepdim=True)
        normed_seqs = (seqs_tensor - mean) / std
        
        # Forecast using Time-MoE
        prediction_length = args.prediction_length
        output_timemoe = model_timemoe.generate(normed_seqs, max_new_tokens=prediction_length)
        normed_predictions = output_timemoe[:, -prediction_length:].cpu().numpy().reshape(prediction_length, -1)
        
        predictions = (normed_predictions * std.cpu().numpy()) + mean.cpu().numpy()
        logging.info("Time-MoE forecast generated successfully.")
        
        # Prepare the forecast index
        last_index = df.index[-1]
        if hasattr(last_index, 'tzinfo') and last_index.tzinfo is not None:
            if args.utc:
                last_index = last_index  # Keep timezone awareness if utc is True
            else:
                last_index = last_index.tz_convert(None)  # Convert timezone-aware index to naive if necessary
        forecast_index = pd.date_range(start=last_index + pd.Timedelta(1, unit=frequency), periods=prediction_length, freq=frequency)

        # Save Time-MoE results to DataFrame
        result_df_timemoe = pd.DataFrame(predictions, index=forecast_index, columns=["prediction_timemoe"])

        # Save to CSV
        output_path = args.output if args.output else "timemoe_forecast_results.csv"
        result_df_timemoe.to_csv(output_path, index=True)
        logging.info(f"Time-MoE prediction results saved to {output_path}")

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Time-MoE Forecast on a given time series dataset.")
    parser.add_argument("--path", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--column", type=str, required=True, help="Name of the column to be predicted.")
    parser.add_argument("--utc", type=bool, required=True, help="Whether the dataframe is timezone-aware.")
    parser.add_argument("--prediction_length", type=int, default=48, help="Length of the prediction.")
    parser.add_argument("--context_length", type=int, default=168, help="Length of the context.")
    parser.add_argument("--output", type=str, help="Path to save the output CSV file.")
    parser.add_argument("--frequency", type=str, help="Frequency of the time series data (e.g., 'H' for hourly). If not provided, it will be inferred.")

    args = parser.parse_args()
    main(args)