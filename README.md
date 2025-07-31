# Customer Churn Prediction using Amazon SageMaker Pipelines

## Overview
This project builds and automates an end-to-end ML pipeline using SageMaker Pipelines for predicting customer churn. The model uses XGBoost, hyperparameter tuning, Clarify explainability, and batch transform.

## Pipeline Components
- **Data preprocessing** using SKLearnProcessor
- **Training and HPO** using XGBoost and TuningStep
- **Model Evaluation** using a custom script
- **Model Registration** for deployment
- **Model Explainability** using SageMaker Clarify
- **Batch Prediction** using TransformStep

## Final Clarify Output
SHAP report and feature importance graphs can be found under `reports/`.

## How to Reproduce
1. Upload data to your S3 bucket: `s3://your-bucket/data/...`
2. Run `SageMaker_Pipelines_Project.ipynb` or trigger from Studio UI

## Notes on Pipeline Execution

While the full SageMaker pipeline was implemented successfully, the final `TransformStep` failed to execute due to an AWS service quota limit on the selected instance type. A service quota increase request has been submitted (screenshot attached). 

To avoid further delay, the project is being submitted with:
- Screenshots of the pipeline structure and steps
- A previously successful Clarify explainability run
- Confirmation that all other pipeline steps (preprocessing, training, evaluation, registration) executed successfully

This approach ensures transparency while allowing progress to continue on remaining assignments.
