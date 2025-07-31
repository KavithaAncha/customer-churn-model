import os
import json
import tarfile
import subprocess
import sys
import pandas as pd

# Install xgboost 
try:
    import xgboost as xgb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.6.2"])
    import xgboost as xgb

# Load test data
print("Loading test data from: /opt/ml/processing/test/test_final.csv")
test_df = pd.read_csv("/opt/ml/processing/test/test_final.csv")
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# Unpack model tarball
model_dir = "/opt/ml/processing/model"
model_tar_path = os.path.join(model_dir, "model.tar.gz")
print(f"Looking for model file in: {model_dir}")

with tarfile.open(model_tar_path, "r:gz") as tar:
    print("=== Files inside model.tar.gz ===")
    for member in tar.getmembers():
        print(member.name)
    tar.extractall(model_dir)

# Locate the model file 
model_file_path = None
for root, dirs, files in os.walk(model_dir):
    for file in files:
        if file == "xgboost-model" or file.endswith(".json") or file.endswith(".bst"):
            model_file_path = os.path.join(root, file)
            break

if model_file_path is None:
    raise ValueError("No valid XGBoost model file found in extracted tarball.")

# Load model
print(f"Loading model from: {model_file_path}")
model = xgb.Booster()
model.load_model(model_file_path)

# Evaluate model
dtest = xgb.DMatrix(X_test, label=y_test)
preds = model.predict(dtest)
pred_labels = (preds >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

report_dict = {
    "classification_metrics": {
        "accuracy": accuracy_score(y_test, pred_labels),
        "precision": precision_score(y_test, pred_labels),
        "recall": recall_score(y_test, pred_labels),
        "auc_score": {
            "value": roc_auc_score(y_test, preds),
        }
    }
}

# Save evaluation report
output_dir = "/opt/ml/processing/evaluation"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "evaluation.json")

with open(output_path, "w") as f:
    json.dump(report_dict, f)

print(f"Saved evaluation report to: {output_path}")
