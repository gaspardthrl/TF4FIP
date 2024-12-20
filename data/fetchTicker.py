import argparse
import yfinance as yf
import pandas as pd

def fetch(ticker='ES=F', frequency="1h", session_start=None, session_end=None, tz=None):

    # Fetch historical data for the specific trading session at 1-minute intervals
    data = yf.download(tickers=ticker, interval=frequency, start=session_start, end=session_end, prepost=False)

    # Set the correct timezone and account for daylight saving hours
    if frequency[-1] in ["m", "h"] and tz is not None:
      data.index = data.index.tz_convert(tz)
    # Reset index to get a clean DataFrame
    data.reset_index(inplace=True)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Ticker")
    parser.add_argument("--ticker", type=str, required=True, help="Name of the ticker")
    parser.add_argument("--frequency", type=str,choices=['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'], required=True, help="Frequency of the fetched data")
    parser.add_argument("--session_start", type=str, required=True, help="Start of the trading session.")
    parser.add_argument("--session_end", type=str, required=True, help="End of the trading session")
    parser.add_argument("--tz", type=str, default=None, help="Timezone (Applicable if frequency is intra-day)")
    parser.add_argument("--output_path", type=str, default="fetched_data.csv", help="The name of the destination file.")
    args = parser.parse_args()

    if args.tz is not None:
       session_start = pd.Timestamp(args.session_start, tz=args.tz)
       session_end = pd.Timestamp(args.session_end, tz=args.tz)
    else:
       session_start = pd.Timestamp(args.session_start)
       session_end = pd.Timestamp(args.session_end)
    
    df = fetch(ticker=args.ticker, frequency=args.frequency, session_start=session_start, session_end=session_end, tz=args.tz)
    df.columns = df.columns.droplevel("Ticker")
    df.to_csv(args.output_path)