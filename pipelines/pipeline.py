from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformInput
from sagemaker.clarify import SageMakerClarifyProcessor
import sagemaker
import os


def get_pipeline(role, default_bucket, pipeline_name="ChurnPredictionPipeline"):
    pipeline_session = sagemaker.workflow.pipeline_context.PipelineSession()

    # === Training Step ===
    xgb_estimator = XGBoost(
        entry_point="pipelines/customerchurn/train.py",
        framework_version="1.5-1",
        hyperparameters={"objective": "binary:logistic", "num_round": 100},
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
    )

    step_train = TrainingStep(
        name="TrainChurnModel",
        estimator=xgb_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{default_bucket}/data/train/train_final.csv",
                content_type="text/csv"
            )
        },
    )

    # === Create Model Step ===
    model = Model(
        image_uri=sagemaker.image_uris.retrieve("xgboost", region=pipeline_session.boto_region_name, version="1.5-1"),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=pipeline_session,
    )

    step_create_model = ModelStep(
        name="CreateModelStep",
        step_args=model.create(instance_type="ml.m5.xlarge")
    )

    # === Evaluation Step ===
    sklearn_eval_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=pipeline_session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    step_eval = ProcessingStep(
        name="EvaluateModelStep",
        processor=sklearn_eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/data/test/test_final.csv",
                destination="/opt/ml/processing/test"
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation"
            )
        ],
        code="pipelines/customerchurn/evaluate.py",
        property_files=[evaluation_report]
    )

    # === Batch Transform Step ===
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type="ml.t3.medium",
        output_path=f"s3://{default_bucket}/batch-transform-output",
        sagemaker_session=pipeline_session,
    )

    step_transform = TransformStep(
        name="BatchTransformStep",
        transformer=transformer,
        inputs=TransformInput(
            data=f"s3://{default_bucket}/data/test/test_final.csv",
            content_type="text/csv"
        )
    )

    # === Clarify Step ===
    clarify_processor = SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
    )

    clarify_script_processor = ScriptProcessor(
        image_uri=clarify_processor.image_uri,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=pipeline_session,
    )

    step_clarify = ProcessingStep(
        name="ClarifyExplainability",
        processor=clarify_script_processor,
        code="pipelines/customerchurn/run_clarify_cleaned.py",
        job_arguments=[
            "--model_name", step_create_model.properties.ModelName,
            "--default_bucket", default_bucket,
            "--clarify_output_path", f"s3://{default_bucket}/clarify-output/",
            "--clarify_instance_type", "ml.m5.xlarge",
            "--clarify_instance_count", "1",
            "--label", "label",
            "--threshold", "0.5",
        ]
    )

    # === Model Registration ===
    step_register = RegisterModel(
        name="RegisterChurnModel",
        estimator=xgb_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="ChurnModelPackageGroup"
    )

    # === Condition Step ===
    cond_gte = ConditionGreaterThan(
        left=JsonGet(
            step_name="EvaluateModelStep",
            property_file=evaluation_report,
            json_path="classification_metrics.auc_score.value"
        ),
        right=0.8
    )

    step_cond = ConditionStep(
        name="CheckEvaluationAndRegister",
        conditions=[cond_gte],
        if_steps=[step_register, step_transform, step_clarify],
        else_steps=[]
    )

    # === Final Pipeline ===
    return Pipeline(
        name=pipeline_name,
        steps=[step_train, step_create_model, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
