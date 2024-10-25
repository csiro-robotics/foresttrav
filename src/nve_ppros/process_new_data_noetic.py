from pathlib import Path
import sys

from tqdm import tqdm

from core.data_preprocessing import reindexes_bags
from core.data_preprocessing import generate_cropped_ply
from core.wildcat_preprocessing import run_wildcat_pipeline

# ROS noetic and wildcat and ohm

# This file does the following
# 1. Cleans up all the directories
# 2. Run wildcat on all the relevant bags
# 3. Extract the processed ply for {odom, global} if available


ROOT_NEW_DATA_DIR = "/data/data_sets/2021_09_20"

PACK_NAME = "p5"
ROBOT_INDENT = "squash"
TRAJ_TYPES = ["global"]
PCL_CROPPED_PADDING = 3.0


def main(argv):
    root_dir_name = ROOT_NEW_DATA_DIR

    if len(argv) > 1:
        print(Path(argv[1]))
        root_dir_name = argv[1]
    data_path = Path(root_dir_name)

    wildcat_dirs = []
    rc_dirs = []
    print("Reminder to the lazy and skimmers of readmes:")
    print("1) IS THERE A ROSCORE RUNNING?")
    print("2) IS WILDCAT SOURCED?")

    run_preprocessing(data_path)


def run_preprocessing(root_new_data_dir: Path) -> None:
    """ """
    for data_dir in tqdm(root_new_data_dir.iterdir()):

        if not data_dir.exists() or not data_dir.is_dir():
            continue

        reindexes_bags(bag_dir=data_dir)

    for data_dir in tqdm(root_new_data_dir.iterdir()):

        if not is_wc_dir(data_dir):
            continue

        run_wildcat_pipeline(
            data_dir=data_dir, pack_ident=PACK_NAME, robot_ident=ROBOT_INDENT
        )

    for data_dir in tqdm(root_new_data_dir.iterdir()):
        for traj_types in TRAJ_TYPES:
            generate_cropped_ply(
                dir=data_dir, traj_type=traj_types, padding=PCL_CROPPED_PADDING
            )


def is_rc_dir(data_dir: Path) -> list:
    """Check if this is the rc directory"""
    return list(data_dir.glob("*_safety.yaml"))


def is_wc_dir(data_dir: Path) -> list:
    """Check if this a wc directory"""
    return list(data_dir.glob("*agent.yaml"))


if __name__ == "__main__":
    main(sys.argv)
