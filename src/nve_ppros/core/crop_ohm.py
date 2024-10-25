from pathlib import Path
import data_preprocessing

import subprocess


def main():

    data_dir_name = "/data/exp_data_processed"
    data_root_dir = Path(data_dir_name)

    for dir in data_root_dir.iterdir():
        if not dir.is_dir():
            continue

        crop_ohm_maps(dir, traj_type="global", ohm_file="tm_map_full.ohm")


def crop_ohm_maps(data_dir: Path, traj_type, ohm_file):

    # Get bounds by trajectory

    # Ohm Map
    res = 0.1
    map_mode = "tm"
    padding = [3, 3, 1]

    map_in = data_dir / f"{traj_type}_{data_dir.name}_{map_mode}_map_v{res}.ohm"
    if not map_in.exists():
        print(f"[ERROR] OHM file does not exists, abborting {map_in}")
        return

    # Generate out file
    ohm_out_file = (
        data_dir
        / f"{traj_type}_{data_dir.name}_{map_mode}_map_v{res}_submap.ohm"
    )

    if ohm_out_file.exists():
        print(f"Skipping {str(data_dir)}, cropped file exists {ohm_out_file}")
        return
    # Trajectory files
    traj_file = data_dir / f"{traj_type}_wildcat_traj.txt"
    if not traj_file.exists():
        print("[ERROR] Trajectory file does not exists, abborting")
        return
    map_dim = data_preprocessing.generate_bounds_from_traj(
        traj_file=traj_file, pad=padding
    )

    # Run ohm cropp
    run_ohm_cropping(in_file=map_in, out_file=ohm_out_file, new_map_dim=map_dim)


def run_ohm_cropping(in_file: Path, out_file: Path, new_map_dim: list) -> None:
    """Runs the ohm submap processing and generates an out file"""

    dims_string = ",".join([str(x) for x in new_map_dim])
    args_pop = f"ohmsubmap -i {in_file} -o {out_file} --box={dims_string}"
    print(args_pop)

    p_out = subprocess.run([args_pop], shell=True, check=True, stdout=subprocess.PIPE)


if __name__ == "__main__":
    main()
