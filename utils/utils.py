import os
import pandas as pd
def save_scaled_data(df_train, df_val, df_test, save_dir="scaled_data"):
    """
    Saves the scaled datasets to disk as Parquet files.

    Parameters:
    - df_train, df_val, df_test: Scaled DataFrames
    - save_dir: Directory where files will be saved
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    df_train.to_parquet(f"{save_dir}/train_scaled.parquet", index=False)
    df_val.to_parquet(f"{save_dir}/val_scaled.parquet", index=False)
    df_test.to_parquet(f"{save_dir}/test_scaled.parquet", index=False)

    print(f"✅ Scaled datasets saved in '{save_dir}'")


def load_scaled_data(save_dir="scaled_data"):
    """
    Loads the previously saved scaled datasets.

    Parameters:
    - save_dir: Directory where files are stored

    Returns:
    - df_train, df_val, df_test (DataFrames)
    """
    df_train = pd.read_parquet(f"{save_dir}/train_scaled.parquet")
    df_val = pd.read_parquet(f"{save_dir}/val_scaled.parquet")
    df_test = pd.read_parquet(f"{save_dir}/test_scaled.parquet")

    print(f"✅ Scaled datasets loaded from '{save_dir}'")

    return df_train, df_val, df_test
