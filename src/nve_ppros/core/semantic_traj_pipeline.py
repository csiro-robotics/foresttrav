import subprocess
import os
from pathlib import Path


def main():
    data_dir = "/data/exp_data_sets/2021_12_13_23_42_55Z"
    traj_type = "global"
    config_file = "/data/data_processed/config_pipline.yaml"

    run_semantic_trajectory_generation(
        data_dir=data_dir, traj_type=traj_type, config_file=config_file
    )


def run_semantic_trajectory_generation(data_dir, traj_type, config_file, debug=0):
    """Class to run semantic trajectory generation"""
    traj_file = os.path.join(data_dir, "{}_wildcat_traj.txt".format(traj_type))

    rc_files = generate_dir_string_array(data_dir, "rc_states")
    out_file = os.path.join(data_dir, "{}_wildcat_traj_labeled.txt".format(traj_type))
    # Generate bash excecutable
    arg_sem_traj_gen_pipeline = "ros2 run nve_lfd traj_labler_node --ros-args "
    arg_sem_traj_gen_pipeline += ' -p "traj_file:={}"'.format(traj_file)
    arg_sem_traj_gen_pipeline += ' -p "rc_bag_files:={}"'.format(rc_files)
    arg_sem_traj_gen_pipeline += ' -p "output_file:={}"'.format(out_file)
    arg_sem_traj_gen_pipeline += ' -p "pipeline:={}"'.format("true")
    arg_sem_traj_gen_pipeline += ' --params-file "{}"'.format(config_file)

    if debug:
        print("Command to run semantic traj generation")
        print(arg_sem_traj_gen_pipeline)

    p1 = subprocess.run(
        [arg_sem_traj_gen_pipeline], shell=True, check=True, stdout=subprocess.PIPE
    )


def generate_file_string_array(data_dir, pattern):
    """Generates a string list of files used for launching ros2
    example : { "[file,file2, file3] " }


    """
    abs_wildcat_patters = os.path.join(data_dir, pattern)
    files_list = glob.glob(abs_wildcat_patters)

    if len(files_list) != 0:
        return ' " [' + ",".join(files_list) + '] "'

    return [""]


def generate_dir_string_array(data_dir, pattern):
    """Generates a string list of files used for launching ros2
    example : { "[file,file2, file3] " }
    Note: Expecting the patterns to be found in dirs...


    Note: NO WILDACARDS ALLOWED
    """
    files = []
    for item in os.listdir(data_dir):

        a = os.path.join(data_dir, item)
        b = pattern in item
        c = os.path.isdir(os.path.join(data_dir, item))
        if pattern in item:
            # if os.path.isdir(os.path.join(abs_path, item)) and pattern in item:
            temp_file = '"' + os.path.join(data_dir, item) + '"'
            files.append(temp_file)

    return "[" + ",".join(files) + "]"


if __name__ == "__main__":
    main()
