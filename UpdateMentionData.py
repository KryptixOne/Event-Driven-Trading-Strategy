import time
import pandas as pd
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import os
from datetime import datetime


def merge_existing_data(existing_csv: str = "./data/wsb_data.csv", extra_excel: str = "./data/wsb-all.xlsx"):
    # Load existing data
    print('Starting Merge')
    if os.path.exists(existing_csv):
        existing_df = pd.read_csv(existing_csv)
    else:
        print("Error: Existing CSV file not found.")
        exit()

    # Load new Excel data (extracting required columns)
    extra_df = pd.read_csv(extra_excel, usecols=["Ticker", "Datetime", "Mentions"])

    # Convert 'Datetime' to the correct format
    extra_df["Datetime"] = pd.to_datetime(extra_df["Datetime"]).dt.strftime("%Y-%m-%d")

    # Rename columns to match existing CSV format
    extra_df.rename(columns={"Datetime": "Date"}, inplace=True)

    # Merge both datasets on Ticker and Date
    merged_df = pd.merge(existing_df, extra_df, on=["Ticker", "Date"], how="outer", suffixes=("", "_extra"))

    # Resolve duplicate columns (if 'Mentions' from different sources exist)
    merged_df["Mentions"] = merged_df[["Mentions", "Mentions_extra"]].max(axis=1)  # Keep the highest value
    merged_df.drop(columns=["Mentions_extra"], inplace=True)

    # Reset MultiIndex if present
    merged_df = merged_df.reset_index(drop=True)

    # Convert 'Mentions' to numeric (fix dtype issues)
    merged_df["Mentions"] = pd.to_numeric(merged_df["Mentions"], errors="coerce").astype(float)

    # Sort by Ticker & Date before calculations
    merged_df = merged_df.sort_values(by=["Ticker", "Date"])

    # Ensure 'Percent Mention Change' exists
    if "Percent Mention Change" not in merged_df.columns:
        merged_df["Percent Mention Change"] = None

    # Forward-fill first (preserves existing values)
    merged_df["Percent Mention Change"] = merged_df.groupby("Ticker")["Percent Mention Change"].transform(
        lambda x: x.ffill())

    # Compute missing values only where NaN
    missing_mask = merged_df["Percent Mention Change"].isna()
    merged_df.loc[missing_mask, "Percent Mention Change"] = (
        merged_df.groupby("Ticker")["Mentions"].transform(lambda x: x.pct_change() * 100)
    )

    merged_df.to_csv('./data/merged_wsb_data_orig.csv', index=False)

    print("Merge completed successfully.")


def get_quiver_and_market_data(csv_file: str = "./data/wsb_data.csv"):
    # Setup Selenium WebDriver
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Load QuiverQuant WSB page
    url = "https://www.quiverquant.com/wallstreetbets/"
    driver.get(url)
    time.sleep(2)

    # Scrape stock tickers and mentions
    stocks = []
    rows = driver.find_elements(By.XPATH, '//table/tbody/tr')

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) > 1:
            ticker = cols[0].text.strip().replace("$", "")
            mentions = int(cols[1].text.strip())
            percent_mention_change = float(cols[2].text.strip().replace(',', '').replace('%', ''))
            timestamp = datetime.now().strftime("%Y-%m-%d")

            stocks.append([timestamp, ticker, mentions, percent_mention_change])

    driver.quit()

    # Convert to DataFrame
    df = pd.DataFrame(stocks, columns=["Date", "Ticker", "Mentions", "Percent Mention Change"])

    # Fetch market data for each ticker
    market_data = []
    for ticker in df["Ticker"]:
        try:
            stock = yf.Ticker(ticker)
            hist_all = stock.history(period="7d")
            if len(hist_all) >= 2:
                hist = hist_all.iloc[-1]  # Most recent day
                seven_day_change = hist_all.iloc[-1] / hist_all.iloc[0]
            else:
                print(f"Not enough History to Proceed for {ticker}")
                continue
            open_price = hist["Open"]
            close_price = hist["Close"]
            volume = hist["Volume"]
            seven_day_change_open = seven_day_change["Open"]
            seven_day_change_close = seven_day_change["Close"]
            seven_day_change_volume = seven_day_change["Volume"]
            market_cap = stock.info.get("marketCap", "N/A")

            market_data.append([ticker, market_cap, open_price, close_price, volume,
                                seven_day_change_open, seven_day_change_close, seven_day_change_volume])
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            market_data.append([ticker, "N/A", "N/A", "N/A"])

    # Convert market data to DataFrame
    market_df = pd.DataFrame(market_data,
                             columns=["Ticker", "Market Cap", "Open Price", "Close Price", "Volume",
                                      "7 Day Change Open", "7 Day Change Close", "7 Day Change Volume"])

    # Merge with original DataFrame
    df = df.merge(market_df, on="Ticker")

    # Check if CSV exists and append new data
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df]).drop_duplicates(subset=["Date", "Ticker"], keep="last")

    # Save updated data
    df.to_csv(csv_file, index=False)

    print("Data updated successfully.")


def fill_missing_financial_data(merged_csv: str, yh_csv = None):
    """
    Reads the merged CSV with missing columns (Market Cap, Open Price, Close Price, etc.),
    fetches historical data for all required dates & tickers from Yahoo Finance,
    and fills the missing columns (including 7-day changes).
    """

    if not os.path.exists(merged_csv):
        print(f"Error: File {merged_csv} not found.")
        return

    df = pd.read_csv(merged_csv, parse_dates=["Date"], dayfirst=False)

    # 2) Identify earliest & latest dates and unique tickers
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    tickers = df["Ticker"].unique().tolist()

    # We'll fetch from (earliest_date - 7 days) to (latest_date + 1 day)
    # so we can properly compute 7-day changes for the earliest date in your file.
    start_date = (min_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if not tickers:
        print("No tickers found in the CSV.")
        return

    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")

    # 3) Bulk download full history for all tickers & date range
    invalid_tickers = []
    valid_tickers = []

    for t in tickers:
        # Convert to string if it's not
        if not isinstance(t, str):
            invalid_tickers.append(t)
            continue
        st = t.strip()
        if not st or st.lower() == "nan":
            invalid_tickers.append(t)
        else:
            valid_tickers.append(st)

    print("Invalid tickers removed:", invalid_tickers)
    if not yh_csv:
        hist_data = batch_download_yf(valid_tickers, start_date, end_date)
    else:
        hist_data = pd.read_csv(yh_csv)

    # Rename columns to match your desired names
    hist_data.rename(
        columns={
            "Date": "Date",
            "Open": "Open Price",
            "Close": "Close Price",
            "Volume": "Volume",
        },
        inplace=True,
    )

    hist_data.sort_values(by=["Ticker", "Date"], inplace=True)

    # Group by ticker, then compute 7-day changes for Open/Close/Volume
    # (row's value / row's value from 7 days earlier)
    def _seven_day_ratio(series):
        return series / series.shift(7)

    hist_data["7 Day Change Open"] = hist_data.groupby("Ticker")["Open Price"].transform(_seven_day_ratio)
    hist_data["7 Day Change Close"] = hist_data.groupby("Ticker")["Close Price"].transform(_seven_day_ratio)
    hist_data["7 Day Change Volume"] = hist_data.groupby("Ticker")["Volume"].transform(_seven_day_ratio)

    df["Date"] = pd.to_datetime(df["Date"])
    hist_data["Date"] = pd.to_datetime(hist_data["Date"])
    # 7) Merge back into df (left join ensures we keep all existing rows in df)
    columns_to_merge = [
        "Date",
        "Ticker",
        "Open Price",
        "Close Price",
        "Volume",
        "7 Day Change Open",
        "7 Day Change Close",
        "7 Day Change Volume",
    ]

    common_tickers = set(df["Ticker"]).intersection(set(hist_data["Ticker"]))

    # 🔹 Step 2: Filter both datasets to keep only common tickers
    df = df[df["Ticker"].isin(common_tickers)].copy()
    hist_data = hist_data[hist_data["Ticker"].isin(common_tickers)].copy()

    df_merged = pd.merge(
        df,
        hist_data[columns_to_merge],
        on=["Date", "Ticker"],
        how="outer",
        suffixes=("", "_hist"),
    )

    # 8) Fill missing data in the main columns from the newly fetched columns
    #    Example: If 'Open Price' was missing, fill it with 'Open Price_hist'
    for col in [
        "Market Cap",
        "Open Price",
        "Close Price",
        "Volume",
        "7 Day Change Open",
        "7 Day Change Close",
        "7 Day Change Volume",
    ]:
        hist_col = col + "_hist"
        if hist_col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(df_merged[hist_col])
            df_merged.drop(columns=[hist_col], inplace=True, errors="ignore")
    df_merged = df_merged.dropna(subset=['Close Price'])
    df_merged = df_merged.dropna(subset=['Volume'])
    df_merged["Mentions"] = df_merged["Mentions"].fillna(0)

    # Define your threshold
    threshold = 800
    row_counts = df_merged["Ticker"].value_counts()
    # Get tickers with row counts >= threshold
    tickers_to_keep = row_counts[row_counts >= threshold].index

    # Filter df_merged to keep only those tickers
    df_merged = df_merged[df_merged["Ticker"].isin(tickers_to_keep)]

    # 9) Save the updated CSV
    df_merged.sort_values(by=["Ticker", "Date"], inplace=True)
    df_merged.to_csv('final_csv.csv', index=False)
    print(f"Missing financial data filled successfully in final_csv.csv!")


def batch_download_yf(valid_tickers, start_date, end_date, batch_size=500):
    """
    Downloads historical market data in batches to avoid rate limits.

    Args:
        valid_tickers (list): List of stock tickers.
        start_date (str): Start date in format 'YYYY-MM-DD'.
        end_date (str): End date in format 'YYYY-MM-DD'.
        batch_size (int, optional): Number of tickers per batch (default: 100).

    Returns:
        pd.DataFrame: Merged historical market data.
    """

    all_data = []

    for i in range(0, len(valid_tickers), batch_size):
        batch = valid_tickers[i:i + batch_size]  # Get the next batch
        print(f"Fetching batch {i // batch_size + 1}/{(len(valid_tickers) // batch_size) + 1}...")

        try:
            hist_data = yf.download(batch, start=start_date, end=end_date, group_by="ticker", threads=True)
            if isinstance(hist_data.columns, pd.MultiIndex):
                hist_data = hist_data.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index()

            all_data.append(hist_data)

        except Exception as e:
            print(f"Error downloading batch {i // batch_size + 1}: {e}")

        time.sleep(300)  # Rate limit buffer

    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    return final_df


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_colwidth', 20)

    csv_file_path = "./data/wsb_data_orig.csv"
    merged_csv = './data/merged_wsb_data_orig.csv'
    data_from_quiver = "./data/wsb-all.csv"
    # How to run
    #get_quiver_and_market_data(csv_file=csv_file_path)
    #merge_existing_data(existing_csv=csv_file_path, extra_excel=data_from_quiver)
    fill_missing_financial_data(merged_csv=merged_csv,yh_csv ="./data/after_yh.csv" )
