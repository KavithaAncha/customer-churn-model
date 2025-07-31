import pandas as pd
import xgboost as xgb
import os
import joblib

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    input_path = "/opt/ml/input/data/train"
    output_path = "/opt/ml/model"
    os.makedirs(output_path, exist_ok=True)

    # load train_final.csv instead of train.csv
    df = pd.read_csv(os.path.join(input_path, "train_final.csv"))

    X = df.drop(columns=["label"])
    y = df["label"]

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "eta": 0.2
    }

    bst = xgb.train(params, dtrain, num_boost_round=100)

    model_path = os.path.join(output_path, "xgboost-model")
    bst.save_model(model_path)