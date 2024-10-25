#!/usr/bin/env python

# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

import copy
import time
from pathlib import Path

import numpy as np
import rospy
from joblib import load
from matplotlib import pyplot as plt
from sensor_msgs.msg import PointCloud2

import ros_numpy as rnp
from odap_tm.models.BinaryTEEnsemble import BinaryFtmEnsemble, parse_iter_model_from
from ros_scnn_tm.msg import TriggerModelUpdate
from scnn_tm.utils.costmap_converter import costmap_conversion
import tf2_ros


class ScnnFtmRosNode:
    """Class to estimate traversability on a per voxel basis"""

    def __init__(self):
        self.te_pub = rospy.Publisher("~tep_cloud", PointCloud2, queue_size=1)

        self.load_ros_params()

        if not self.model_dir:
            msg = "[ForestTravNode]: Model or feature set files not defined"
            rospy.logwarn(msg)

        # Load model
        self.model_configs = parse_iter_model_from(
            self.model_dir, self.cv_start, self.cv_number
        )
        self.scnn_model = None

        # Used to collect the header
        self.data_ = []
        self.msg_frame_id = ""
        self.msg_time_stamp = 0
        self.received_new_data_ = False  # Flag to check if new data has been received

        self.online_model_update = rospy.Subscriber(
            "~online_model_update", TriggerModelUpdate, self.update_model_configs
        )
        self.subscription_ = rospy.Subscriber(
            "~input_feature_cloud", PointCloud2, self.pcl_cb
        )

        if self.publish_costmap:
            self.te_costmap_pub = rospy.Publisher(
                "~te_costmap_cloud", PointCloud2, queue_size=1
            )
            self.tf_buffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def load_ros_params(self):
        self.model_dir = rospy.get_param("~model_dir", "")
        self.cv_start = rospy.get_param("~cv_start", 0)
        self.cv_number = rospy.get_param("~cv_number", 10)
        self.device = rospy.get_param("~device", "gpu")

        # Used to visualize the cloud in warm colours: "bwr_r", "viridis", "coolwarm_r"
        self.visualise_colour = rospy.get_param("~visualise_colour", False)
        self.colour_map = rospy.get_param("~clour_map", "viridis")
        self.cmap = plt.colormaps[self.colour_map]

        # Static member variable used for incremental learning
        self.update_model_trigger = False

        # Costmap pub
        self.publish_costmap = rospy.get_param("~costmap_conversion", 0)
        self.bounds = [-3.0, -3.0, -3.0, 3.0, 3.0, 3.0]

    def pcl_cb(self, msg):
        """Callback to process the received msg"""
        self.data_ = rnp.numpify(msg)
        # self.msg_frame_id = msg.header.frame_id
        self.msg_frame_id = msg.header.frame_id
        self.msg_time_stamp = msg.header.stamp

        if not self.scnn_model:
            self.input_features = [
                field.name
                for field in msg.fields
                if field.name not in ["x", "y", "z", "rgb"]
            ]

            # Hacky way to deal with colour
            if "red" in self.input_features:
                self.input_colour_idx = [
                    self.input_features.index("red"),
                    self.input_features.index("green"),
                    self.input_features.index("blue"),
                ]
                self.has_colour = True
            else:
                self.has_colour = False

        # Trigger to enable main cycle on next loop
        self.received_new_data_ = True

    def update_model_configs(self, msg):
        """Callback to trigger the model param update

        Args:
            - update model file <not necessary in first step>
            - update cv_start, necessary
            - update cv_number <not necessary at start>
        """
        print("Updating models")
        self.cv_start = msg.cv_start
        self.cv_number = msg.cv_number

        if not Path(msg.file_path).is_dir():
            rospy.logwarn(f"Model file does not exists {msg.file_path}")
            return

        self.model_dir = Path(msg.file_path)
        self.model_configs = parse_iter_model_from(
            self.model_dir, self.cv_start, self.cv_number
        )

        if not self.model_configs:
            rospy.logwarn(f"No models found in:{msg.file_path}")
            return

        self.scnn_model = None

    def load_models(self):
        if not self.model_configs:
            rospy.logdebug_throttle(30.0, "No valid model found for training")
            return
        rospy.loginfo("[SCNN ROS] Loading new models")
        rospy.loginfo("Loading new models")

        self.scnn_model = BinaryFtmEnsemble(self.model_configs, self.device, self.input_features)

    def main_cb(self, event=None) -> ModuleNotFoundError:
        """Main callback to call all the process to enable the predictions data"""

        # Should we process?
        if not self.received_new_data_ or len(self.data_) < 1:
            rospy.logdebug_throttle(30.0, "Did not receive any new data")
            return

        # Need the input_features so the model can be initialized
        if not self.scnn_model:
            self.load_models()

            if not self.scnn_model:
                rospy.logdebug_throttle(30.0, "Could not load a new model")
                return

        # This is the most senseless and ugly conversion in a long while
        # Called a structured array in numpy and needs to be converted with a view
        start_time = time.perf_counter()

        X_coords = []
        feature_data = []
        # If we need to do the conversion
        start_0 = time.perf_counter()
        data_org = copy.deepcopy(self.data_)

        dtypes_coord = [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
        ]
        X_coords = (
            data_org[["x", "y", "z"]]
            .astype(dtypes_coord)
            .view("<f4")
            .reshape(data_org.shape[0], len(dtypes_coord))
        )

        dtypes_feature = [(f_i, "<f4") for f_i in self.input_features]
        feature_data = (
            data_org[self.input_features]
            .astype(dtypes_feature)
            .view("<f4")
            .reshape(data_org.shape[0], len(dtypes_feature))
        )

        # Deal
        if self.has_colour:
            feature_data[:, self.input_colour_idx] = (
                feature_data[:, self.input_colour_idx] / 255.0
            )

        start_1 = time.perf_counter()

        # Prediction
        prediction_arr, pred_var = self.scnn_model.fast_predict(
            X_coords=X_coords, X_features=feature_data, voxel_size=0.1
        )
        start_2 = time.perf_counter()

        #
        data_out_type = []

        if self.visualise_colour:
            data_out_type = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("prob", "<f4"),
                    ("var", "<f4"),
                    ("r", np.uint8),
                    ("g", np.uint8),
                    ("b", np.uint8),
                ]
            )
        else:
            data_out_type = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("prob", "<f4"),
                    ("var", "<f4"),
                ]
            )

        # There must be a more efficient way of doing this
        pred_arr = np.empty(len(X_coords), dtype=data_out_type)
        pred_arr["x"] = X_coords[:, 0].view("f4").reshape(X_coords.shape[0])
        pred_arr["y"] = X_coords[:, 1].view("f4").reshape(X_coords.shape[0])
        pred_arr["z"] = X_coords[:, 2].view("f4").reshape(X_coords.shape[0])

        pred_arr["prob"] = prediction_arr.astype("f4")
        pred_arr["var"] = pred_var.astype("f4")

        colours = []
        if self.visualise_colour:
            cmap = self.cmap
            colours = (cmap(prediction_arr)[:, 0:3] * 255).astype(np.uint)
            pred_arr["r"] = colours[:, 0]
            pred_arr["g"] = colours[:, 1]
            pred_arr["b"] = colours[:, 2]

        start_3 = time.perf_counter()
        self.publish_predictions(pred_arr, self.visualise_colour)
        start_4 = time.perf_counter()

        if self.publish_costmap:
            self.generate_costmap_pcl(X_coords, y_pred=prediction_arr)

        start_5 = time.perf_counter()

        # Counters for time
        print(f"Elapsed time data conversion:  {  start_0 - start_time}")
        print(f"Elapsed time  ohmcloud_conversion:  {  start_1 - start_0}")
        print(f"Elapsed time  prediction:  {  start_2 - start_1}")
        print(f"Elapsed time  prediction conversion :  {  start_3 - start_2}")
        print(f"Elapsed time  publishing:  {  start_4 - start_3}")
        print(f"Elapsed time  costmap conversion:  {  start_5 - start_4}")

        received_new_data = False
        self.data_ = []

    def publish_predictions(self, pred_arr, has_rgb):
        if has_rgb:
            pred_arr = rnp.point_cloud2.merge_rgb_fields(pred_arr)
        msg = rnp.msgify(PointCloud2, pred_arr)
        msg.header.frame_id = self.msg_frame_id
        msg.header.stamp = self.msg_time_stamp
        self.te_pub.publish(msg)
        # print("Publishing prediction")

    def get_translation(self):
        try:
            # Get the transform from "map" to your robot's frame
            transform = self.tf_buffer.lookup_transform(
                "odom", "base_link", rospy.Time(0)
            )

            # Get the translational part of the transform
            translation = transform.transform.translation
            return  np.array([translation.x, translation.y, translation.z])

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):

            rospy.logdebug_throttle(30,"TF lookup failed. Waiting for TF data...")

            return np.empty((0,3))

    def generate_costmap_pcl(self, X_coords, y_pred):

        # Get local bounfs
        pos_t = self.get_translation()

        # Failed position lookup, abort
        if not pos_t.shape[0]:
            return

        # Generate costmap
        costmap = costmap_conversion(
            point_cloud=np.hstack([X_coords, y_pred.reshape(-1, 1)]),
            voxel_size=0.1,
            voxels_per_collum=10,
        )

        data_out_type = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("prob", "<f4"),
                ("r", np.uint8),
                ("g", np.uint8),
                ("b", np.uint8),
            ]
        )

        costmap_pcl = np.empty(costmap.shape[0], dtype=data_out_type)
        costmap_pcl["x"] = costmap[:, 0].reshape(costmap.shape[0])
        costmap_pcl["y"] = costmap[:, 1].reshape(costmap.shape[0])
        costmap_pcl["z"] = costmap[:, 2].reshape(costmap.shape[0])
        costmap_pcl["prob"] = costmap[:, 3].reshape(costmap.shape[0])

        colours = []

        cmap = self.cmap
        colours = (cmap(costmap[:, 3])[:, 0:3] * 255).astype(np.uint)
        costmap_pcl["r"] = colours[:, 0]
        costmap_pcl["g"] = colours[:, 1]
        costmap_pcl["b"] = colours[:, 2]

        costmap_pcl = rnp.point_cloud2.merge_rgb_fields(costmap_pcl)
        msg = rnp.msgify(PointCloud2, costmap_pcl)
        msg.header.frame_id = self.msg_frame_id
        msg.header.stamp = self.msg_time_stamp
        self.te_costmap_pub.publish(msg)


def main(args=None):
    rospy.init_node("scnn_ftm_ros")

    node = ScnnFtmRosNode()

    rospy.Timer(rospy.Duration(0.1), node.main_cb)

    rospy.spin()


if __name__ == "__main__":
    main()
