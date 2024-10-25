import yaml
import numpy as np
import scnn_tm
from scnn_tm.models.ScnnFtmEstimators import ScnnFtmEnsemble, parse_cv_models_from_dir

#
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    matthews_corrcoef,
    roc_curve,
)


def eval_uncertanty():
    model_dir = "/data/scnn_models_23_03_27/test_train_lfe_hl_10_UNet5LMCD_ndt"
    cv_model_files = parse_cv_models_from_dir(model_dir, 10)

    # Load ScnnEnsemble
    model = ScnnFtmEnsemble(model_config_files=cv_model_files, device="cuda", input_feature_set=scnn_tm.utils.generate_feature_set_from_key("ftm"))

    data_set_name = scnn_tm.utils.test_data_set_by_key("lfe_hl", 0.1)
    data_set = scnn_tm.utils.load_te_data_set(data_set_name[0])
    X_coords = data_set[["x", "y", "z"]].to_numpy()
    X_features = data_set[scnn_tm.utils.generate_feature_set_from_key("ftm")].to_numpy()
    y_true = data_set["label"].to_numpy()

    y_pred, y_var = model.predict(
        X_coords=X_coords,
        X_features=X_features,
        voxel_size=0.1,
        )

    # Plot
    sample_weight = np.ones(y_pred.shape)

    # Calculate the log likelihood
    log_loss
    brier_score = brier_score_loss(y_true=y_true, y_prob=y_pred)
    print("Log likelihood:", log_loss(y_true=y_true, y_pred=y_pred))
    print("Brier_score: ", brier_score)
    print("ROC AUC score", roc_auc_score(y_true=y_true, y_score=y_pred))
    roc_curve(y_true=y_true, y_score=y_pred)

    scnn_tm.utils.visualize_probability(cloud_coords=X_coords, cloud_prob=y_pred)
    scnn_tm.utils.visualize_classification_difference(
        cloud_coords=X_coords, source_labels=y_pred, target_labels=y_true
    )


eval_uncertanty()
