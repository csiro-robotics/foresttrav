import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, matthews_corrcoef
from tabulate import tabulate

from joblib import load
import yaml
from scnn_tm.utils import (
    test_data_set_by_key,
    load_te_data_set,
    visualize_classification_difference,
    train_data_set_by_key,
    data_set_name_without_nte,
    classification_comparsion,
    CLASIIFICATION_TO_CR_LABEL
)
from scnn_tm.models.ScnnFtmEstimators import (
    ScnnFtmEstimator,
    load_cv_unet_model_from_yaml,
)
import copy


def parse_svm_model(model_dir: Path, model_name: str, cv_nbr: int):
    model_dir = Path(model_dir)
    estimator_path = model_dir / (f"{model_name}estimator_cv_{cv_nbr}.joblib")
    estimator = load(estimator_path)

    feature_set_path = model_dir / (f"{model_name}features_cv_{cv_nbr}.joblib")
    feature_set = load(feature_set_path)

    scaler_path = model_dir / (f"{model_name}scaler_cv_{cv_nbr}.joblib")
    scaler = load(scaler_path)

    return estimator, scaler, feature_set


def calculate_score_for_cv_run(y_pred, y_target):

    # Macro scores for the classification
    report_ = classification_report(y_pred=y_pred, y_true=y_target, output_dict=True)
    scores = np.array(
        [
            matthews_corrcoef(y_pred=y_pred, y_true=y_target),
            report_["macro avg"]["f1-score"],
            report_["macro avg"]["precision"],
            report_["macro avg"]["recall"],
        ]
    )
    return scores


def process_scnn_cv_scene(model_params, data_set_key: str):

    # data_sets = train_data_set_by_key(data_set_key, model_params["voxel_size"])
    wandb_config = yaml.safe_load(open( Path(model_params["model_dir"]).parent / "config.yaml","r"))
    data_sets = wandb_config["param/data_sets"]["value"]
    scores = []
    for cv_nbr in range(len(data_sets)):

        test_data_set_file = data_sets[cv_nbr]
        if Path(test_data_set_file).name in data_set_name_without_nte():
            print(f"Skipping data set {data_sets[cv_nbr]} ")
            continue
        
        test_data_set = load_te_data_set(test_data_set_file)

        # Load the model
        try:
            estimator = ScnnFtmEstimator(
                model_file=model_params["model_dir"], device="cuda", cv_nbr=cv_nbr
            )
        except:
            print(f"Failed to load  {cv_nbr} with data set {data_sets[cv_nbr]} ")
            raise ValueError()
        # Get predictions
        X_coords = test_data_set[["x", "y", "z"]].to_numpy()
        X_features = test_data_set[estimator.feature_set].to_numpy()
        y_pred, _ = estimator.predict(
            X_coords=X_coords,
            X_features=X_features,
            voxel_size=model_params["voxel_size"],
        )

        # Calculate metrics
        y_target = test_data_set["label"].to_numpy()
        scores.append(calculate_score_for_cv_run(y_pred, y_target))

    # Stack metrics
    return generate_cv_scores(model_params=model_params, scores=scores)

def process_scnn_test_train(model_params, test_data_set: pd.DataFrame):

    # Check if it is the test_train case:
    scores = []
    for cv_nbr in range(model_params["cv_total"]):
        # Load the model
        try:
            estimator = ScnnFtmEstimator(
                model_file=model_params["model_dir"], device="cuda", cv_nbr=cv_nbr
            )
        except:
            continue
        # Get predictions
        X_coords = test_data_set[["x", "y", "z"]].to_numpy()
        X_features = test_data_set[estimator.feature_set].to_numpy()
        y_pred, _ = estimator.predict(
            X_coords=X_coords,
            X_features=X_features,
            voxel_size=model_params["voxel_size"],
        )

        # Calculate metrics
        y_target = test_data_set["label"].to_numpy()
        scores.append(calculate_score_for_cv_run(y_pred, y_target))

    # Stack metrics
    if True:
        # visualize_classification_difference(cloud_coords=X_coords, source_labels=y_pred, target_labels=y_target)

    
    return generate_cv_scores(model_params=model_params, scores=scores)


def process_svm_cv(
    model_params: dict,
    test_data_set: pd.DataFrame,
):
    """Processes the score for the a model using svm and cross validation"""

    scores = []
    y_pred = []
    y_target = []
    X_coords = []
    for i in range(model_params["cv_total"]):

        # Load the model
        estimator, scaler, feature_set = parse_svm_model(
            model_dir=model_params["model_dir"],
            model_name=model_params["model_name"],
            cv_nbr=i,
        )

        # Get predictions
        X_coords = test_data_set[["x", "y", "z"]].to_numpy()
        y_pred = estimator.predict(
            scaler.transform(test_data_set[feature_set].to_numpy())
        )
        y_proba = estimator.predict_proba(scaler.transform(test_data_set[feature_set].to_numpy())
        )
        # Calculate metrics
        y_target = test_data_set["label"].to_numpy()
        scores.append(calculate_score_for_cv_run(y_pred, y_target))

    # Stack metrics
    if False:
        a = 1
        visualize_classification_difference(cloud_coords=X_coords, source_labels=y_pred, target_labels=y_target)

        
    # Save images for
    if True:
        
        te_diff = np.array([ CLASIIFICATION_TO_CR_LABEL[classification_comparsion(y_pred[i], y_target[i])] for i in range(len(y_pred)) ])                                          

        out_dir_path = Path("/data/scnn_models_23_04_01/predicted_images")
        voxel_size = model_params["voxel_size"]
        model_name=model_params["model_name"] + f"_v{voxel_size}"
        out_file = out_dir_path / ( f"test_scene _{ model_name }.csv" )
        
        df_out = test_data_set.copy()
        df_out.insert(4, "te_diff", te_diff)
        df_out.insert(4, "y_proba", y_proba[:,1])
        df_out.insert(5, "y_pred", y_pred)
        df_out.to_csv(out_file, index=False, sep=",")
    
    return generate_cv_scores(model_params=model_params, scores=scores)


def generate_cv_scores(model_params, scores):
    """Returns a list such that it contains all the information's to generate the final report
    [model_name estimator_type feature_set_key mcc_mean mcc_std f1_mean f1_std prec_mean prec_st recall_mean recall_std ]

    """
    stacked_scores = np.vstack(scores)
    return [
        model_params["model_name"],
        model_params["model_type"],
        model_params["data_set_key"],
        model_params["feature_set_key"],
        np.round(np.mean(stacked_scores[:, 0]), 4),
        np.round(np.std(stacked_scores[:, 0]), 4),
        np.round(np.mean(stacked_scores[:, 1]), 4),
        np.round(np.std(stacked_scores[:, 1]), 4),
        np.round(np.mean(stacked_scores[:, 2]), 4),
        np.round(np.std(stacked_scores[:, 2]), 4),
        np.round(np.mean(stacked_scores[:, 3]), 4),
        np.round(np.std(stacked_scores[:, 3]), 4),
    ]


def main():

    # Definition for tabulate
    header = [
        "data_key",
        "model_name",
        "estimator_type",
        "feature_set",
        "mcc_mean",
        "mcc_std",
        "f1_mean",
        "f1_std",
        "prec_mean",
        "prec_std",
        "recall_mean",
        "recall_std",
    ]
    data_set_keys = ["lfe_hl"]
    voxel_size = 0.1
    total_scores = []

    # Load yaml config file
    model_configs = yaml.safe_load(
        open("/nve_ml_ws/src/scnn_tm/config/test_train_model_eval_v1_cl.yaml", "r")
    )

    for data_set_key in data_set_keys:
        test_data_set_dirs = test_data_set_by_key(data_set_key, voxel_size)
        test_data_set = load_te_data_set(test_data_set_dirs[0])

        for key, model_params in model_configs.items():
            #
            if model_params["model_type"] == "svm":
                total_scores.append(
                    [data_set_key] + process_svm_cv(model_params, test_data_set)
                )

            elif model_params["model_type"] == "scnn" and model_params["data_set_key"] == "cv_scene":
                total_scores.append(
                    [data_set_key] + process_scnn_cv_scene(model_params, data_set_key)
                )
            elif model_params["model_type"] == "scnn" and model_params["data_set_key"] == "test_train":
                total_scores.append(
                    [data_set_key] + process_scnn_test_train(model_params, test_data_set)
                )

            else:
                raise ValueError()

    # Add
    print(tabulate(total_scores))


if __name__ == "__main__":
    main()
