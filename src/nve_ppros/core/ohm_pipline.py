from dataclasses import dataclass
import subprocess
from pathlib import Path

# OHM PROCESSING PARAMERTERS
RESOLUTIONS = [0.1]
MODE = "TM"
TRAJ_TYPE = "global"
RETURN_NUMBER_TYPE = "explicit"
MAX_RAY_LENGTH = "20.0"
VERBOSE = True

# MAPPING PARAMETERS

DATA_DIRS = [
    # "/data/data_sets/2021_10_28/2021_10_28_04_03_11Z",
    # "/data/data_sets/2021_10_28/2021_10_28_04_13_12Z",
    # "/data/data_sets/2021_11_18/2021_11_18_00_23_24Z",
    # "/data/data_sets/2021_12_13/2021_12_13_23_51_33Z",
    # "/data/data_sets/2021_12_14/2021_12_14_00_02_12Z",
    # "/data/data_sets/2021_12_14/2021_12_14_00_14_53Z",
    # "/data/data_sets/2022_02_14/2022_02_14_23_47_33Z",
    # "/data/data_sets/2021_12_13/2021_12_13_23_42_55Z",
    # "/data/data_sets/2022_02_14/2022_02_15_00_00_35Z",
    "/data/data_sets/2021_10_28/2021_10_28_03_54_49Z",
]
OUT_DIR = "/data/processed/"


def main():
    resolutions = RESOLUTIONS
    data_dirs = DATA_DIRS

    for dir in data_dirs:

        if not Path(dir).is_dir():
            continue
        
        # Define the outdir
        out_dir = Path(OUT_DIR) / Path(dir).stem / Path("ohm_maps")
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        for res in resolutions:

            run_ohm_pipeline(
                data_dir=Path(dir),
                out_dir=Path(out_dir),
                mode="tm",
                resolution=res,
                traj_type="global",
                return_number="explicit",
                traversal_enable=True,
                max_ray_length=MAX_RAY_LENGTH,
                verbose=True
            )

    print("Finished all files")


def run_ohm_pipeline(
    data_dir: Path,
    out_dir: Path,
    mode: str, 
    resolution: float, 
    traj_type: str, 
    return_number: str,
    traversal_enable: bool,
    max_ray_length: float,
    verbose: bool = True
):
    """
    Will compute all ohm maps in a certain directory for all named resolution

    Args:
        data_dir:           Path to the data file
        out_dir:            Output directory
        mode:               Mode of the occupanyc map {"tm"}
        resolutions:        Resolution of map voxels in meters
        traversal_enable:   Enables the traversal of the map
        traj_type:          Type of trajectory to use {"odom", "global"}
        return_number:      Return number to use {"explicit", "none"}
        max_ray_length:     Maximum ray length in meters
        verbose:

    """

    # Generate the out files
    out_file = out_dir / f"{traj_type}_{data_dir.name}_{mode}_map_v{resolution}"
    traj_file, ply_file = get_traj_and_ply_file(data_dir=data_dir, traj_type=traj_type)

    if verbose:
        print(f"Start processsing {data_dir.absolute} with resolutions {resolution}")

    args_pop = f"ohmpopcpu {ply_file} \
            {traj_file} \
            {out_file} \
            --ndt={mode} \
            --resolution={resolution} \
            --traversal={traversal_enable}\
            --return_number={return_number} \
            --ray-length-max={max_ray_length} \
            --trace-final=0"

    print(args_pop)
    # Run the subprocess using the target dier
    p_out = subprocess.run(
        [args_pop], shell=True, check=True, stdout=subprocess.PIPE, cwd=data_dir
    )



def get_traj_and_ply_file(data_dir: Path, traj_type:str):
    """ Finds the trajectory and ply file in the data directory given a trajectory type

    Args:
        data_dir:   Path to the data directory
        traj_type:  Type of trajectory to use {"odom", "global"}
    
    Returns:
        traj_file:  Path to the trajectory file
        ply_file:   Path to the ply file

    """
    # Generte the the input files
    ply_file = data_dir.joinpath(f"{traj_type}_wildcat_velodyne.ply")
    traj_file = data_dir.joinpath(f"{traj_type}_wildcat_traj.txt")
    
    if not ply_file.exists() or not traj_file.exists():
        print(f"[OHM PIPELINE] Failed to find a ply or traj file in the directory {data_dir}")
        print(f"[OHM PIPLINE] Target ply file: {ply_file}")
        print(f"[OHM PIPLINE] Target trajectory file: {traj_file}")
        return Path(""), Path("")

    return traj_file, ply_file

if __name__ == "__main__":
    main()
