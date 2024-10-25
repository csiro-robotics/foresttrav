import argparse
import pandas as pd
from pathlib import Path
from joblib import load
import logging
import numpy as np
import time
import scnn_tm
from scnn_tm.models.ScnnFtmEstimators import ScnnFtmEnsemble, parse_cv_models_from_dir


from scnn_tm.utils.visualisations import (
    classification_comparsion,
    visualize_probability,
    visualize_classification_difference, 
    CLASIIFICATION_TO_CR_LABEL, 
)
from scnn_tm.utils.scnn_io import (
    generate_feature_set_from_key,
)

SCENE_FILES = [
    "/data/processed/feature_sets/lfe_hl_v0.1/2022_02_14_23_47_33Z",
    # "/data/processed/feature_sets/23_08_02_fix_ev/lfe_hl_v0.1/2022_02_14_23_47_33Z",
    # "/data/processed/feature_sets/lfe_hl_v0.1/2021_12_14_00_02_12Z",
    # "/data/processed/ohm_scans/2022_02_14_23_47_33Z/ohm_scans_v01/1644882496970399472_cloud.csv",
    # "/data/processed/ohm_scans/2022_02_14_23_47_33Z/ohm_scans_v01/1644882535274470225_cloud.csv",
    # "/data/processed/ohm_scans/2022_02_14_23_47_33Z/ohm_scans_v01/1644882600644600927_cloud.csv",
]
USE_SCNN = False
# UNET_MODEL_FILE = "/data/forest_trav_paper/ts_models/cv_test_train_lfe_hl_10_UNet5LMCD_ohm_mr/model_config.yaml"
# SCNN_ENS_DIR = "/data/forest_trav_paper/ts_models/cv_test_train_lfe_hl_10_UNet5LMCD_ohm_mr"

MODEL_CV_CONFIG = [
    # "/data/ts_scnn_models/23_05_30_occ_cv_scene",
    # "/data/experiments/2023_08_03/wandb/run-20230802_233610-23_08_02_23_36_UNet5LMCD_s3_nf16_ftm_d0.1_cv_test_train_ADAM/files",
    # "/data/experiments/2023_08_03/wandb/run-20230802_230802-23_08_02_23_08_UNet5LMCD_s3_nf16_ohm_mr_d0.1_cv_test_train_ADAM/files",
    "/data/forest_trav_paper/ts_models/cv_test_train_lfe_hl_10_UNet5LMCD_ohm_mr",
    # "/data/experiments/2023_08_07/wandb/run-20230809_223722-23_08_09_22_37_UNet5LMCD_s3_nf16_ohm_mr_d0.1_cv_test_train_ADAM/files"
    ]


# MODEL_CV_CONFIG = [
# "/data/scnn_models_23_04_01/models_hl/test_train_lfe_hl_10_UNet5LMCD_occ",
# "/data/scnn_models_23_04_01/models_hl/test_train_lfe_hl_10_UNet5LMCD_occ_rgb",
# "/data/scnn_models_23_04_01/models_hl/test_train_lfe_hl_10_UNet5LMCD_occ_perm_int",
# "/data/scnn_models_23_04_01/models_hl/test_train_lfe_hl_10_UNet5LMCD_occ_perm_int_ev_sr",
# "/data/scnn_models_23_04_01/models_hl/test_train_lfe_hl_10_UNet5LMCD_ndt",
# "/data/scnn_models_23_04_01/models_hl/test_train_lfe_hl_10_UNet5LMCD_ftm",
# ]

# MODEL_CV_CONFIG = [
# "/data/scnn_models_23_04_01/models_hl/cv_patch_lfe_hl_10_UNet5LMCD_occ",
# "/data/scnn_models_23_04_01/models_hl/cv_patch_lfe_hl_10_UNet5LMCD_occ_rgb",
# "/data/scnn_models_23_04_01/models_hl/cv_patch_lfe_hl_10_UNet5LMCD_occ_perm_int",
# "/data/scnn_models_23_04_01/models_hl/cv_patch_lfe_hl_10_UNet5LMCD_occ_perm_int_ev_sr",
# "/data/scnn_models_23_04_01/models_hl/cv_patch_lfe_hl_10_UNet5LMCD_ndt",
# "/data/scnn_models_23_04_01/models_hl/cv_patch_lfe_hl_10_UNet5LMCD_ftm",
# ]

def main(args,):

    # Load model
    
    for i in range(len(MODEL_CV_CONFIG)):
        iterate_over_model(args, i)

def iterate_over_model(args, i = -1):
    model_dir = MODEL_CV_CONFIG[i]
    model_config_files = parse_cv_models_from_dir(model_dir, 10)
    model_name = Path(model_dir).stem
    
    # Setup out dir if required'
    if args.save_prediction:
        out_dir_path = Path(args.out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

    # Define what feature set we expect
    model_feature_set = generate_feature_set_from_key("ohm_mr")
    model_ensemble = ScnnFtmEnsemble(model_config_files, "cuda", model_feature_set)

    for scene in args.scene_files:
        df = []
        if Path(scene).is_file():
            df = pd.read_csv(scene)
        elif Path(scene).is_dir():
            df_0 = pd.read_csv(Path(scene) / "semantic_cloud_class_1.csv")
            df_0["label"] = 0.0
            df_1 = pd.read_csv(Path(scene) / "semantic_cloud_class_2.csv")
            df_1["label"] = 1.0
            df = pd.concat([df_0, df_1])

        # X_features = scaler.transform(df[feature_set].to_numpy())
        X_coord = df[["x", "y", "z"]].to_numpy()
        X_feature = df[model_feature_set].to_numpy()
        y_target = df["label"].to_numpy()
        time_0 = time.perf_counter()
        # Load scene
        pred_mean, pred_var = model_ensemble.predict(
            X_coords=X_coord, X_features=X_feature, voxel_size=0.1
        )
        time_1 = time.perf_counter()
        pred_binary  = np.zeros(pred_mean.shape)
        pred_binary[pred_mean > 0.5]  = 1.0
        print(f"Ensemble took {time_1-time_0}")


        
        if args.save_prediction:
            te_diff = np.array([ CLASIIFICATION_TO_CR_LABEL[classification_comparsion(pred_binary[i], y_target[i])] for i in range(len(pred_binary)) ])
            out_file = out_dir_path / ( f"{Path(scene).name}_{model_name}.csv" )
            # df_out = df[["x", "y", "z"]]
            df["y_pred"] = pred_mean
            df["y_pred_label"] = pred_binary
            df["pred_var"] = pred_var
            df["te_diff"] = te_diff
            df["y_true"] = y_target
            df.to_csv(out_file, index=False, sep=",")

        if args.visualize:
            visualize_classification_difference(cloud_coords=X_coord, source_labels=pred_binary, target_labels=y_target)
            visualize_probability(cloud_coords=X_coord, cloud_prob=pred_mean)

parser = argparse.ArgumentParser()

parser.add_argument("--scene_files", required=False, type=list, default=SCENE_FILES)
parser.add_argument( "--out_dir", required=False, type=str, default="/data/forest_trav/predicted_scenes/test_scene_0.1")
parser.add_argument("--save_prediction", required=False, type=bool, default=True)
parser.add_argument("--visualize", required=False, type=bool, default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
