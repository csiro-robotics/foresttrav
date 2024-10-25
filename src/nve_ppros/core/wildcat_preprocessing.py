# Author: Fabio Ruetz

# @brief: Classess and functions to automate the wildcat preprocessing pipeline
from pathlib import Path
import subprocess
import pandas as pd


# Runs the wildcat pipeline
#  @p[in]: Root work directory for the data set
#  - Reindexes all bags in the given folder
#  - Adds the bags to the wildcat yaml
#  - Processes wildcat and ouput

ROOT_NEW_DATA_DIR = ["/data/new_data_23_04_06/2023_04_06_04_15_05Z",]
PACK_IDENT = "p18"
ROBOT_IDENT = "rat"

OVERRIDE = False


def main():

    # Labeling pipeline

    for data_dir in ROOT_NEW_DATA_DIR:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            continue

        run_wildcat_pipeline(data_dir, pack_ident=PACK_IDENT, robot_ident=ROBOT_IDENT)


def run_wildcat_pipeline(data_dir: Path, pack_ident: str, robot_ident: str) -> None:

    if not can_process_wc(data_dir=data_dir, pack_name=pack_ident):
        print(f"[WILDCAT] Could not find wildcat files (pack indent) in {data_dir}")
        return

    if has_been_processed(data_dir=data_dir) and not OVERRIDE:
        print(f"[WILDCAT] Skipping {data_dir} as it has been processed")
        return

    if not update_wildcat_yaml(data_dir=data_dir, pack_ident=pack_ident):
        print(f"[WILDCAT] Failed to update wildcat.yaml . Skipping {data_dir}")
        return

    wc_file_name = "wildcat_mod.yaml"
    self_strike_name = f"{robot_ident}_self_strike.yaml"
    pack_cal_name = f"{pack_ident}_wildcat_cal.yaml"

    run_wildcat(
        data_dir=data_dir,
        wc_file_name=wc_file_name,
        self_strike_name=self_strike_name,
        pack_cal_name=pack_cal_name,
    )


def run_wildcat(
    data_dir: Path, wc_file_name: str, self_strike_name, pack_cal_name
) -> None:
    """ """
    arg_pipeline = f"rosrun wildcat_ros pipeline -p {wc_file_name} {pack_cal_name} {self_strike_name} -o nominalOverrides offlineOverrides -n offlineFull"

    print(arg_pipeline)
    # p1 = subprocess.run(
    #     [arg_pipeline], shell=True, check=True, stdout=subprocess.PIPE, cwd=data_dir
    # )
    print(f"[WILDCAT] Finished running wildcat for data set {data_dir}")


def can_process_wc(data_dir: Path, pack_name: str) -> bool:
    """Check if wildcat can be processed by looking for key config files"""

    pack_config_exists = bool(list(data_dir.glob(f"{PACK_IDENT}_*.bag")))
    wc_config_exists = bool(list(data_dir.glob(f"*wildcat.yaml")))
    sf_config_exists = bool(list(data_dir.glob(f"*_self_strike*")))

    return pack_config_exists and wc_config_exists and sf_config_exists


def has_been_processed(data_dir: Path) -> bool:
    """Check if the wildcat files exist"""
    wc_ply_exists = bool(list(data_dir.glob(f"*wildcat_velodyn*.ply")))
    wc_traj_exists = bool(list(data_dir.glob(f"*wildcat_traj.txt")))

    return wc_ply_exists and wc_traj_exists


def update_wildcat_yaml(data_dir: Path, pack_ident):
    """Updates the wildcat yaml files and generates the wildcat_mod.yaml
    Main reason is to add the bag files to the wildcat.yamls
    Note: it preserves ordering and spacing
    """
    wc_org_file = data_dir.joinpath("wildcat.yaml")

    writer = open(str(wc_org_file)[:-5] + "_mod.yaml", "w+")

    with open(wc_org_file, "r") as reader:
        for line in reader:
            if "INPUT_RAW_BAG_FILES: [" in line:
                # Modify the saving bag

                bag_files = list(data_dir.glob(f"{pack_ident}_*.bag"))
                bag_file_names = [file.name for file in bag_files]

                modified_line = " INPUT_RAW_BAG_FILES: [{files}]".format(
                    files=", ".join(bag_file_names)
                )

                writer.write(modified_line)

            else:
                writer.write(line)

    writer.close()
    print(f"[WILDCAT] Finished updating wildcat.yaml files for data set {data_dir}")
    return True


if __name__ == "__main__":
    main()
