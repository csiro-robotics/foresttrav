# Author: Fabio Ruetz
#
# Set of functions to pre-process files on a folder based structure
from core import plyfile_util
from core import pcl_filters

import pandas as pd
from plyfile import PlyData, PlyElement
import numpy as np
import subprocess
import shutil
from pathlib import Path


def reindexes_bags(bag_dir: Path):
    """
    Will find all active bags in a directory and re-index them if they don't already exists without the active sign
    Note: Requires a roscore to be active AND the ros1 version to be sourced
    """
    no_bags = True
    for file in bag_dir.glob("*bag.active"):
        out_file = bag_dir.joinpath(file.name[:-7])

        if "orig" not in file.name and not bag_dir.joinpath(out_file).exists():

            print("Reindexing rosbag: with file name: ", out_file)

            p1 = subprocess.run(
                ["rosbag reindex {}".format(file)],
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
            )

            p2 = subprocess.run(
                [
                    "mv {org_bag_file} {bag_file}".format(
                        org_bag_file=file, bag_file=out_file
                    )
                ],
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
            )
            print("Finished rosbag reindexing and saved to ", out_file)
            no_bags = False  # flag to print to visualize what we check the dir
        else:
            continue

    if no_bags:
        print("No bag files found to re-index in ", bag_dir)


def convertRosBags(r1_in_file, r2_out_file):

    p1 = subprocess.run(
        [
            "rosbags-convert {bag_file} --dst {out_file}".format(
                bag_file=r1_in_file, out_file=r2_out_file
            )
        ],
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
    )


def copy_files(data_dir, out_dir, pattern): 

    file_names = Path(data_dir).glob(pattern)
    out_base_dir = Path(out_dir)
    for file in list(file_names):

        if not file.is_file():
            continue

        print("File name", Path(file).name)
        shutil.copyfile(file, out_base_dir.joinpath(Path(file).name))



def symlink_files_by_pattern(dir_in:Path, dir_out:Path, pattern: str)->None:
    """ Generates sim links from the data_dir in in the data_dir_out files with the given pattern

    Parameters
    ----------
    data_dir_in : Path
        The input directory where the files are located
    data_dir_pout : Path
        The output directory where the sim links should be created
    pattern : str
        File patterns that should be linked

    Returns
    -------
    None
    """

    file_names = Path(dir_in).glob(pattern)
    out_base_dir = Path(dir_out)
    for file in list(file_names):

        if not file.is_file():
            continue

        print("File name", Path(file).name)
        out_file = out_base_dir.joinpath(Path(file).name)
        
        if not out_file.exists():
            out_file.symlink_to(file)


def generate_cropped_ply(dir: Path, traj_type: str, padding: list, override=False):
    """
    Generates a processed ply file

    """
    bounds = []
    traj_file = dir.joinpath("{}_wildcat_traj.txt".format(traj_type))

    if not traj_file.exists():
        print("[Cropp Ply] No trajectory found, skipping files in ", dir)
        return

    # traj_file = dir.joinpath(dir, "{}_wildcat_traj.txt".format(traj_type))
    df = pd.read_csv(traj_file, skiprows=0, delimiter="\s+")

    bounds.append(df["x"].min() - padding[0])  # x_min
    bounds.append(df["x"].max() + padding[0])  # x_max

    bounds.append(df["y"].min() - padding[1])  # y_min
    bounds.append(df["y"].max() + padding[1])  # y_max

    bounds.append(df["z"].min() - padding[2])  # z_min
    bounds.append(df["z"].max() + padding[2])  # z_max

    # Add padding to each values

    ply_file = dir / (f"{traj_type}_wildcat_velodyne.ply")
    out_ply_file = dir / f"{traj_type}_wildcat_velodyne_processed_.ply"

    if (not ply_file.exists() or out_ply_file.exists()) and not override:
        print("Found a processed ply file and skipping files in ", dir)
        return

    # print("Starte filtering cloud file: ", ply_file)
    plydata = PlyData.read(ply_file)

    cloud_arr = np.zeros(shape=[plydata["vertex"].count, 6], dtype=np.dtype("d"))
    cloud_arr[:, 0] = plydata["vertex"].data["time"]
    cloud_arr[:, 1] = plydata["vertex"].data["x"]
    cloud_arr[:, 2] = plydata["vertex"].data["y"]
    cloud_arr[:, 3] = plydata["vertex"].data["z"]
    cloud_arr[:, 4] = plydata["vertex"].data["intensity"]
    cloud_arr[:, 5] = plydata["vertex"].data["returnNum"]

    cloud_arr = pcl_filters.box_filter(cloud_arr, bounds, 1)

    types_tup = [
        ("time", "f8"),
        ("x", "f8"),
        ("y", "f8"),
        ("z", "f8"),
        ("intensity", "u4"),
        ("returnNum", "u4"),
    ]
    plyfile_util.writeToPlyFile(
        path=out_ply_file, cloud_array=cloud_arr, types_tup=types_tup
    )
    # print("Finished filtering")


def generate_bounds_from_traj(traj_file: Path, pad: list) -> list:

    df = pd.read_csv(traj_file, skiprows=0, delimiter="\s+")

    bounds = []
    if not traj_file.exists:
        print(
            f"Failed to generate new map bounds since traj_file not found: \n {traj_file}"
        )
        return bounds

    bounds.append(df["x"].min() - pad[0])  # x_min
    bounds.append(df["y"].min() - pad[1])  # y_min
    bounds.append(df["z"].min() - pad[2])  # z_min
    bounds.append(df["x"].max() + pad[0])  # x_max
    bounds.append(df["y"].max() + pad[1])  # y_max
    bounds.append(df["z"].max() + pad[2])  # z_max

    return bounds
