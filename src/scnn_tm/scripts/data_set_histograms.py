from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scnn_tm.utils. import generate_feature_set_from_key
from models.FtmScaler import FtmScaler

# FEATURE_SETS = [
#     "/data/processed/feature_sets/2023_02_14_00_27_lfe_hlcl_nve/2021_10_28_03_54_49Z",
#     "/data/processed/feature_sets/2023_02_14_00_27_lfe_hlcl_nve/2021_10_28_04_03_11Z",
#     "/data/processed/feature_sets/2023_02_14_00_27_lfe_hlcl_nve/2021_10_28_04_13_12Z",
#     "/data/processed/feature_sets/2023_02_14_00_27_lfe_hlcl_nve/2021_11_18_00_23_24Z",
#     "/data/processed/feature_sets/2023_02_14_00_27_lfe_hlcl_nve/2021_12_13_23_42_55Z",
#     "/data/processed/feature_sets/2023_02_14_00_27_lfe_hlcl_nve/2021_12_13_23_51_33Z",
#     "/data/processed/feature_sets/2023_02_14_00_27_lfe_hlcl_nve/2021_12_14_00_02_12Z",
#     "/data/processed/feature_sets/2023_02_14_00_27_lfe_hlcl_nve/2021_12_14_00_14_53Z",
    
# ]
FEATURE_SETS = [
    # "/data/processed/scene_feature_sets/lfe_hl/2021_10_28_03_54_49Z",
    # "/data/processed/scene_feature_sets/lfe_hl/2021_10_28_04_03_11Z",
    # "/data/processed/scene_feature_sets/lfe_hl/2021_10_28_04_13_12Z",
    # "/data/processed/scene_feature_sets/lfe_hl/2021_11_18_00_23_24Z",
    # "/data/processed/scene_feature_sets/lfe_hl/2021_12_13_23_42_55Z",
    # "/data/processed/scene_feature_sets/lfe_hl/2021_12_13_23_51_33Z",
    # "/data/processed/scene_feature_sets/lfe_hl/2021_12_14_00_02_12Z",
    # "/data/processed/scene_feature_sets/lfe_hl/2021_12_14_00_14_53Z",
    "/data/processed/feature_sets/2023_02_14_00_30_lfe_hl/2022_02_14_23_47_33Z",
    # "/data/processed/2021_12_14_00_14_53Z/ohm_scans_adj/1639441258.562101_pose_cloud_adj.csv",
    # "/data/processed/2021_12_14_00_14_53Z/ohm_scans_adj/1639441209.366400_pose_cloud_adj.csv",
    # "/data/processed/2021_12_14_00_14_53Z/ohm_scans_adj/1639441151.614778_pose_cloud_adj.csv",
]

SCALER_DICT = {
    "percetile_to_scale": 0.95,
    "occupancy_prob": [0.0, 1.0],
    "permeability": [0.0, 1.0],
    "red": [0.0, 1.0],
    "green": [0.0, 1.0],
    "blue": [0.0, 1.0],
    "theta": [0.0, 1.570796],
}
OUT_FILE = "/data/debug/feature_hist/ohm_scans_feature.pdf"

def load_semantic_classes_from_dir(dir: Path):
    df_0 = pd.read_csv( Path(dir) / "semantic_cloud_class_1.csv")
    df_0["label"]  = 0
        
    df_1 = pd.read_csv(Path(dir)  / "semantic_cloud_class_2.csv")
    df_1["label"]  = 1
    return pd.concat([df_0, df_1])
    

def main(files_list: list):
    
    df_tot_scenes = pd.DataFrame()
    # Load the data
    for data_file in files_list:
        df_new = pd.DataFrame()
        
        if Path(data_file).is_dir():
            df_new = load_semantic_classes_from_dir(data_file)

        if Path(data_file).is_file():
            df_new = pd.read_csv(Path(data_file))
            
        df_tot_scenes = pd.concat([df_tot_scenes,df_new])
    
    feature_set =  scan_io.generate_feature_set_from_key("ftm")
    data_set = df_tot_scenes[feature_set].to_numpy()
    
    scaler  = FtmScaler(0.95, predefined_scaling_bounds=SCALER_DICT)
    scaled_data_set = scaler.fit_scaler(data_set =data_set, feature_set=feature_set, scale_inplace=True)

    # df_tot_scenes["label"] = 0 
    mask_label_0 = df_tot_scenes["label"] < 1
    
    # Save the data as histograms
    figures = []
    out_pdf_file = Path(OUT_FILE)
    out_pdf_file.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf_file) as export_pdf:  
        for i, feature in enumerate(feature_set):
            fig, axs =  plt.subplots(3, sharey=False, tight_layout=True)
            fig.suptitle(f'Feature: [{feature}]')
            axs[0].hist(scaled_data_set[:,i], bins=25)
            axs[0].set_title("Total data set")
            axs[1].hist(scaled_data_set[mask_label_0][:,i], bins=25, color= "red")
            axs[1].set_title("Non-traversable")
            axs[2].hist(scaled_data_set[~mask_label_0][:,i], bins=25,color= "green")
            axs[2].set_title("Traversable")
            export_pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    main(FEATURE_SETS)