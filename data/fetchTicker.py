import argparse

import pandas as pd
import yfinance as yf


def fetch(ticker="ES=F", frequency="1h", session_start=None, session_end=None):

    # Fetch historical data for the specific trading session at 1-hour intervals
    data = yf.download(
        tickers=ticker,
        interval=frequency,
        start=session_start,
        end=session_end,
        prepost=True,
    )

    data.index = data.index.tz_convert('US/Eastern')

    # Reset index to get a clean DataFrame
    data.reset_index(inplace=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Ticker")
    parser.add_argument("--ticker", type=str, required=True, help="Name of the ticker")
    parser.add_argument(
        "--frequency",
        type=str,
        default="1h",
        choices=["1h", "1d"],
        help="Frequency of the fetched data",
    )
    parser.add_argument(
        "--session_start", type=str, required=True, help="Start of the trading session."
    )
    parser.add_argument(
        "--session_end", type=str, required=True, help="End of the trading session"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data.csv",
        help="The name of the destination file.",
    )
    args = parser.parse_args()

    df = fetch(
        ticker=args.ticker,
        frequency=args.frequency,
        session_start=args.session_start,
        session_end=args.session_end,
    )
    df.columns = df.columns.droplevel("Ticker")
    df.to_csv(args.output_path)
