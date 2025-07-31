import subprocess
subprocess.call(["pip", "install", "--upgrade", "sagemaker"])

from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    DataConfig,
    ModelConfig,
    SHAPConfig
)
from sagemaker import Session
import boto3

# ------- Update values -------
role = "arn:aws:iam::819481466570:role/service-role/AmazonSageMaker-ExecutionRole-20250722T173254"
bucket = "sagemaker-us-east-2-819481466570"
model_name = "pipelines-f3dkn74cafe9-CreateModelStep-Crea-syvgjFzTQR"
region = "us-east-2"
clarify_instance_type = "ml.m5.xlarge"
clarify_output_path = f"s3://{bucket}/clarify-output"
train_data_uri = f"s3://{bucket}/data/train/train.csv"

# ------- Feature and label setup -------
headers = [
    "label", "feature_7", "feature_8", "feature_9", "feature_10",
    "feature_11", "feature_12", "feature_13", "feature_14", "feature_15",
    "feature_16", "feature_17", "feature_18", "feature_19", "feature_20", "feature_21"
]
label = "label"

# ------- SageMaker session setup -------
session = Session()
clarify_processor = SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type=clarify_instance_type,
    sagemaker_session=session
)

# ------- Clarify configs -------
data_config = DataConfig(
    s3_data_input_path=train_data_uri,
    s3_output_path=clarify_output_path,
    label=label,
    headers=headers,
    dataset_type="text/csv"
)

model_config = ModelConfig(
    model_name=model_name,
    instance_type=clarify_instance_type,
    instance_count=1,
    accept_type="text/csv"
)

shap_config = SHAPConfig(
    baseline=[["0"] * (len(headers) - 1)],  
    num_samples=100,
    agg_method="mean_abs"
)

# ------- Clarify SHAP explainability job -------
clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=shap_config
)
