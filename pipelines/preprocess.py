import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    input_path = "/opt/ml/processing/input/train.csv"
    print(f"Reading input from: {input_path}")

    try:
        # Assign column names manually 
        columns = [
            "RowNumber", "CustomerId", "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
            "Geography", "Gender", "Exited", "Col1", "Col2", "Col3", "Col4", "Col5",
            "Col6", "Col7", "Col8", "Col9"
        ]
        df = pd.read_csv(input_path, header=None, names=columns)
    except Exception as e:
        print("❌ Failed to read CSV:", str(e))
        raise

    if 'Exited' not in df.columns:
        raise ValueError("❌ 'Exited' column not found in input dataset.")

    # Split data
    try:
        train_df, temp_df = train_test_split(
            df, test_size=0.4, stratify=df["Exited"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["Exited"], random_state=42
        )
    except Exception as e:
        print("❌ Error during train/val/test split:", str(e))
        raise

    # Define output directories
    output_dirs = {
        "train": "/opt/ml/processing/train",
        "validation": "/opt/ml/processing/validation",
        "test": "/opt/ml/processing/test",
    }

    for key, path in output_dirs.items():
        os.makedirs(path, exist_ok=True)

    # Save CSVs
    try:
        train_df.to_csv(f"{output_dirs['train']}/train.csv", index=False)
        val_df.to_csv(f"{output_dirs['validation']}/validation.csv", index=False)
        test_df.to_csv(f"{output_dirs['test']}/test.csv", index=False)
        print("✅ Preprocessing completed and files saved.")
    except Exception as e:
        print("❌ Failed to write output files:", str(e))
        raise
