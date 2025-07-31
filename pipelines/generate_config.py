import json
import os

def generate_analysis_config(output_dir, model_name, default_bucket):
    analysis_config = {
        "dataset_type": "text/csv",
        "headers": ["target", "esent", "eopenrate", "eclickrate", "avgorder", "ordfreq", "paperless", "refill", "doorstep", 
                    "first_last_days_diff", "created_first_days_diff", "favday_Friday", "favday_Monday", "favday_Saturday", 
                    "favday_Sunday", "favday_Thursday", "favday_Tuesday", "favday_Wednesday", "city_BLR", "city_BOM", 
                    "city_DEL", "city_MAA"],
        "label": "target",
        "methods": {
            "shap": {
                "num_samples": 100,
                "agg_method": "mean_abs"
            }
        }
    }

    file_path = os.path.join(output_dir, f"{model_name}_analysis_config.json")
    with open(file_path, "w") as f:
        json.dump(analysis_config, f, indent=4)

    return file_path
