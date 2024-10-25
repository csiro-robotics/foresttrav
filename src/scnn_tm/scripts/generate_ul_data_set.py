import pandas as pd 
import numpy as np
from pathlib import Path

data_sets_root = Path('/data/forest_trav/lfe_hl_v0.1')
new_out_dir = Path('/data/forest_trav/lfe_ul_v0.1')
new_out_dir.mkdir(parents=True, exist_ok=True)

for data_set in data_sets_root.iterdir():
    
    data_set_path = Path(data_set)
    
    # Load the data set
    df = pd.read_csv(data_set_path)
    mask = df["label_obs"] < 1
    df.loc[mask, 'label'] = -1.0
    df.loc[mask, 'label_prob'] = -1.0

    # Save the data in the new location
    new_file = new_out_dir / data_set_path.name
    print(new_file)
    df.to_csv(new_file, index=False)
