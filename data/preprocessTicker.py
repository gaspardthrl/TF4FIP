import numpy as np
import pywt
import sys
import argparse
import pandas as pd

def wavelet_denoise(df, wavelet='db1', level=1):
    """
    Apply wavelet denoising to all numeric columns of a Pandas DataFrame.

    Parameters:
    - df: Pandas DataFrame to denoise
    - wavelet: Type of wavelet to use (default is 'db1')
    - level: Level of decomposition for wavelet transform

    Returns:
    - A DataFrame with denoised data.
    """
    def denoise_column(column):
        coeffs = pywt.wavedec(column, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745 
        threshold = sigma * np.sqrt(2 * np.log(len(column)))
        denoised_coeffs = [pywt.threshold(c, threshold) if i > 0 else c for i, c in enumerate(coeffs)]
        return pywt.waverec(denoised_coeffs, wavelet)[:len(column)]

    denoised_data = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        denoised_data[col] = denoise_column(df[col].values)

    denoised_df = df.copy()
    for col, denoised_col in denoised_data.items():
        denoised_df[col] = denoised_col

    return denoised_df

def preprocess(df, dfDaily=None):
  df.set_index("Datetime", inplace=True)
  df.index = pd.to_datetime(df.index)

  df = df.asfreq('h')
  df = df.interpolate(method="time")

  df["Week"] = df.index.isocalendar().week

  df["Day"] = df.index.dayofweek

  df["Hour"] = df.index.hour

  def emroc(df, column, span, lag):
    df[f"ROC_{column}"] = df[column].pct_change(lag)
    df[f"EMROC_{column}"] = df[f"ROC_{column}"].ewm(span=span, adjust=False).mean()
    return df

  def atr(df, column, span, lag):
    df['Prev Close'] = df['Close'].shift(lag)
    high_low = df['High'] - df['Low']
    high_prev_close = (df['High'] - df['Prev Close']).abs()
    low_prev_close = (df['Low'] - df['Prev Close']).abs()
    df['True Range'] = high_low.combine(high_prev_close, max).combine(low_prev_close, max)
    df[f'ATR_{span}'] = df['True Range'].rolling(window=span).mean()
    df[f'ATR_{span}'] = df[f'ATR_{span}'] / df['Close']
    return df

  def rsi(df, column, span, lag):
      df['Price Change'] = df['Close'].diff()
      df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
      df['Loss'] = df['Price Change'].apply(lambda x: -x if x < 0 else 0)
      df['Avg Gain'] = df['Gain'].rolling(window=span, min_periods=lag).mean()
      df['Avg Loss'] = df['Loss'].rolling(window=span, min_periods=lag).mean()

      df['RS'] = df['Avg Gain'] / df['Avg Loss']

      df['RSI'] = 100 - (100 / (1 + df['RS']))
      return df

  def distancesToMM(df, column, spans):
    for span in spans:
        df[f'MM_{span}'] = df['Close'].rolling(window=span).mean()
        df[f'DistanceToMM{span}'] = ((df["Close"] - df[f'MM_{span}']) / df[f'MM_{span}']) * 100
    return df

  def distancesToEMM(df, column, spans):
      for span in spans:
          df[f'EMM_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
          df[f'DistanceToEMM{span}'] = ((df["Close"] - df[f'EMM_{span}']) / df[f'EMM_{span}']) * 100
      return df

  df = emroc(df, "Close", 72, 2)
  df = emroc(df, "Volume", 72, 2)

  if dfDaily is not None:
    dfDaily["Datetime"] = pd.to_datetime(dfDaily["Date"]).dt.normalize()
    dfDaily.drop(["Date"], axis=1, inplace=True)
    dfDaily.set_index("Datetime", inplace=True)

    dfDaily = dfDaily.asfreq('D')
    dfDaily = dfDaily.interpolate(method="time")

    dfDaily = atr(dfDaily, "Close", 10, 1)
    dfDaily = rsi(dfDaily, "Close", 14, 1)

    dfDaily = distancesToMM(dfDaily, "Close", [20, 60])
    dfDaily = distancesToEMM(dfDaily, "Close", [20, 60])

    row_idx = 0
    rowDaily_idx = 0

    df["ATR_10"] = None
    df["MM_20"] = None
    df["MM_60"] = None
    df["EMM_20"] = None
    df["EMM_60"] = None
    df["RSI"] = None

    step = len(df) / 100
    curr = 1

    def progress_bar(completion, total):
        percent = (completion / total) * 100
        bar = '*' * int(percent // 2)
        bar = bar.ljust(50, ' ')
        if completion == total:
            sys.stdout.write(f"\r[{bar}] 1 of 1 completed\n")
        else:
            sys.stdout.write(f"\r[{bar}] {percent:.0f}%")
        sys.stdout.flush()
    while row_idx < len(df):
        if row_idx >= curr * step:
            progress_bar(curr, 100)
            curr += 1

        if (df.index[row_idx].dayofweek == dfDaily.index[rowDaily_idx].dayofweek and df.index[row_idx].hour >= 17) or ((df.index[row_idx].dayofweek-1)%7 == dfDaily.index[rowDaily_idx].dayofweek and df.index[row_idx].hour < 17):
            # Set values for columns VolumeDiff, CloseDiff, ATR10, MM20, MM60 using .loc
            df.loc[df.index[row_idx], "ATR_10"] = dfDaily.loc[dfDaily.index[rowDaily_idx], "ATR_10"]
            df.loc[df.index[row_idx], "MM_20"] = dfDaily.loc[dfDaily.index[rowDaily_idx], "MM_20"]
            df.loc[df.index[row_idx], "MM_60"] = dfDaily.loc[dfDaily.index[rowDaily_idx], "MM_60"]
            df.loc[df.index[row_idx], "EMM_20"] = dfDaily.loc[dfDaily.index[rowDaily_idx], "EMM_20"]
            df.loc[df.index[row_idx], "EMM_60"] = dfDaily.loc[dfDaily.index[rowDaily_idx], "EMM_60"]
            df.loc[df.index[row_idx], "RSI"] = dfDaily.loc[dfDaily.index[rowDaily_idx], "RSI"]

            row_idx += 1
        else:
            rowDaily_idx += 1
    progress_bar(curr, 100)
    df["DistanceToMM20"] = ((df["Close"] - df["MM_20"]) / df["MM_20"]) * 100
    df["DistanceToMM60"] = ((df["Close"] - df["MM_60"]) / df["MM_60"]) * 100
    df["DistanceToEMM20"] = ((df["Close"] - df["EMM_20"]) / df["EMM_20"]) * 100
    df["DistanceToEMM60"] = ((df["Close"] - df["EMM_60"]) / df["EMM_60"]) * 100

    df["Close_denoised"] = wavelet_denoise(df.copy())["Close"]
    print(df.isna().sum())
    df = df.tail(-max(df.isna().sum()))
    return df
  else:
    return df
def main():
    parser = argparse.ArgumentParser(description='Preprocess financial data with configurable parameters.')
    parser.add_argument('--input_path', help='Path to input CSV file')
    parser.add_argument('--daily_input', help='Path to daily data CSV file', default=None)
    parser.add_argument('--output_path', help='Path to save processed data', default='processed_data.csv')
    parser.add_argument('--config', help='Path to JSON configuration file', default=None)
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_path, index_col=0)
        daily_df = pd.read_csv(args.daily_input, index_col=0) if args.daily_input else None
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    try:
        processed_df = preprocess(df, daily_df)
        processed_df.to_csv(args.output_path)
        print(f"Successfully processed data and saved to {args.output_path}")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()