from pathlib import Path

ROOT_DATA_DIRS = [
    # "/data/data_sets/2021_10_28/2021_10_28_04_03_11Z",
    # "/data/data_sets/2021_10_28/2021_10_28_04_13_12Z",
    # "/data/data_sets/2021_11_18/2021_11_18_00_23_24Z",
    # "/data/data_sets/2021_12_13/2021_12_13_23_51_33Z",
    # "/data/data_sets/2021_12_14/2021_12_14_00_02_12Z",
    # "/data/data_sets/2021_12_14/2021_12_14_00_14_53Z",
    # "/data/data_sets/2022_02_14/2022_02_14_23_47_33Z",
    # "/data/data_sets/2021_12_13/2021_12_13_23_42_55Z",
    "/data/new_data_23_04_06/2023_04_06_04_15_05Z"
]

OUT_DIR = "/data/processed/"


def main(data_dir_list: list, out_sim_dir: Path):

    for dir_name in data_dir_list:
        generate_sim_link_to_raw_data( Path(dir_name), out_sim_dir)




def generate_sim_link_to_raw_data(data_dir: Path, out_sim_dir: Path):
        
        if not data_dir.is_dir() or not data_dir.exists():
            print(f"Element does not exist or is not a directory: {dir}")
            return
        
        new_sim_dir_root = out_sim_dir / data_dir.name 

        if not new_sim_dir_root.exists():
            new_sim_dir_root.mkdir(parents=True)
        
        new_sim_dir = new_sim_dir_root / "raw_data"

        if new_sim_dir.exists():
            print(f"Sim link already exists: {new_sim_dir}")
            return
        # Generate a sim link in de the desried dir
        print(new_sim_dir)
        new_sim_dir.symlink_to(target=data_dir, target_is_directory=True)





if __name__ == "__main__":
    main(ROOT_DATA_DIRS, Path(OUT_DIR))