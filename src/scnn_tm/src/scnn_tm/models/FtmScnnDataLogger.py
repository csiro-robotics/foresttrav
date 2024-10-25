# Copyright (c) 2024
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz


from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import yaml
from tabulate import tabulate


class FtmScnnDataLogger:
    def __init__(self, log_parent_dir: Path, is_root_dir=True) -> None:
        if type(log_parent_dir) != Path:
            log_parent_dir = Path(log_parent_dir)

        if is_root_dir:
            self.log_dir = log_parent_dir / (
                "exp_pl_" + pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")
            )
        else:
            self.log_dir = log_parent_dir

        self.log_file = self.log_dir / "experiment_results.txt"
        self.model_config_file = self.log_dir / "model_config.yaml"

        if not self.log_file.parent.exists():
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_new_model(self, model_name: str, model_features: list):
        """S"""
        f = open(self.log_file, "a+")
        f.write(
            f"\n############################# \n#####    {model_name}    #### \n############################# \n"
        )
        f.write(f"features: {model_features} \n")
        f.close()

    def log_config(self, exp_params: dict):
        f = open(self.log_file, "a+")
        f.write("Config used in this experiments: \n")
        for key, value in exp_params.items():
            f.write(f"Param[ {key} ] = {value} \n")
        f.close()
        print("Saved experiment configuration")

    def log_args(self, args: object):
        config_params = vars(args)
        self.log_config(config_params)

    def log_cv_train_scores(self, model_name, scores):
        f = open(self.log_file, "a+")
        f.write(f"\n CV classification summary scores for mode: {model_name}: \n")
        for key, value in scores.items():
            f.write(f"Param[ {key} ]  mean: {value[0]} std: {value[1]} \n")

        f.write("\n")
        f.close()

    def write_classification_results(
        self, model_name, classification_report, mcc_score
    ):
        f = open(self.log_file, "a+")
        f.write(f"\n \n Classification report for mode: {model_name}: \n")
        f.write(classification_report)
        f.write(f"\n \n Mathews-coefficient: {mcc_score} \n \n")
        f.close()

    def save_torch_model(self, model, cv_number=None, feature_set_key=None) -> None:
        model_name = model.__class__.__name__ + ".pl"
        if not cv_number is None:
            model_name = model.__class__.__name__ + f"_cv_{cv_number}.pl"

        model_file_path = self.log_dir / model_name

        torch.save(model.state_dict(), model_file_path)

        if self.model_config_file.exists():
            return None

        # Create and save data to yaml file
        self.model_config_file.touch()
        with open(self.model_config_file, "w") as file_descriptor:
            yaml_params = {}
            yaml_params["model_name"] = model.__class__.__name__
            yaml_params["model_nfeature_enc_ch_out"] = model.ENCODER_CH_OUT[0]
            yaml_params[
                "model_skip_connection_key"
            ] = f"s{model.SKIP_CONNECTION.count(1)}"
            yaml_params["model_dropout"] = 0
            yaml_params["feature_set_key"] = feature_set_key

            # We make the assumption that the model is in the same dir
            yaml_params["torch_model_path"] = str(model_name)
            yaml_params["scaler_path"] = str((f"scaler_cv_{cv_number}.joblib"))
            yaml.safe_dump(yaml_params, file_descriptor)

    def save_cv_scores(self, scores, header):
        f = open(self.log_file, "a+")
        f.write(f"\n \n Final report over for stacked cv's \n")
        f.write(tabulate([scores], headers=header))
        f.close()

    def final_classification_per_cv_fold(self, y_pred_l, y_label_l):
        f = open(self.log_file, "a+")
        f.write(f"\n \n Final classification report per fold \n \n")
        scores, header = generate_cv_fold_report(y_pred_l=y_pred_l, y_label_l=y_label_l)
        f.write(tabulate(scores, headers=header))
        f.close()


def test_data_logger():
    y_pred_l = [np.random.binomial(n=1, p=0.5, size=(10, 1)) for n in range(10)]
    y_label_l = [np.random.binomial(n=1, p=0.5, size=(10, 1)) for n in range(10)]
    scores, header = generate_cv_fold_report(y_pred_l=y_pred_l, y_label_l=y_label_l)
    print((tabulate(scores, headers=header)))
    # Generate the header


def generate_cv_fold_report(y_pred_l: list, y_label_l: list):
    """Generates a report for n runs of classification, normally cross validation"""
    header = [" "] + [f"Fold: {i}" for i in range(len(y_pred_l))] + ["mean", "std"]

    # Get the cv reports per fold from the classificaion results
    cv_reports = [
        sklearn.metrics.classification_report(
            y_pred=y_pred_l[i], y_true=y_label_l[i], output_dict=True
        )
        for i in range(len(y_label_l))
    ]

    scores = []
    for key in ["f1-score", "precision", "recall"]:
        key_scores = [
            round(cv_reports[i]["macro avg"][key], 4)
            for i in range(len(cv_reports))
            if key != "support"
        ]
        mean_score = round(np.mean(key_scores), 4)
        std_score = round(np.std(key_scores), 4)
        scores.append([key] + key_scores + [mean_score] + [std_score])

    # Add mcc scores
    mcc_score = [
        round(
            sklearn.metrics.matthews_corrcoef(y_pred=y_pred_l[i], y_true=y_label_l[i]),
            4,
        )
        for i in range(len(y_label_l))
    ]
    mcc_mean = round(np.mean(mcc_score), 4)
    mcc_std = round(np.std(mcc_score), 4)
    scores.append(["mcc_score"] + mcc_score + [mcc_mean] + [mcc_std])

    return scores, header
