from pathlib import Path

from scnn_tm.utils import generate_feature_set_from_key
from scnn_tm.utils.point_cloud_utils import convert_ohm_cloud_to_feature_cloud
from scnn_tm.models.ScnnFtmEstimators import ScnnFtmEnsemble, parse_cv_models_from_dir

import ros_numpy as rnp
import numpy as np
import torch

import rospy
from joblib import load
from sensor_msgs.msg import PointCloud2
from matplotlib import pyplot as plt

import time
import copy


class ScnnFtmRosNode:
    """Class to estimate traversability on a per voxel basis"""

    def __init__(self):

        self.do_ohm_to_ftm_conversion = False
        self.subscription_ = rospy.Subscriber("/r6/ohm", PointCloud2, self.pcl_cb)
        self.te_pub = rospy.Publisher("tep_cloud_out", PointCloud2, queue_size=1)

        self.model_file = rospy.get_param(
            "~model_dir", "/data/me_scnn_models/23_05_30_ohm_rgb_cv_patch")
        self.cv_number = rospy.get_param("~cv_number", 10)
        if not self.model_file:
            msg = "[NodeName]: Model or feature set files not defined"
            rospy.logfatal(msg)
            return None
        self.device = rospy.get_param("~device", "cuda")    # "cuda" or "cpu"

        # Load model
        self.model_configs = parse_cv_models_from_dir(self.model_file,  self.cv_number)
        self.scnn_model = None

        # Used to collect the header
        self.data_ = []
        self.msg_frame_id = ""
        self.msg_time_stamp = 0
        self.received_new_data_ = False  # Flag to check if new data has been received
        self.reset_feature_set_mapping = True # Flag to check the incoming feature sets


    def pcl_cb(self, msg):
        """Callback to process the received msg"""
        self.data_ = rnp.numpify(msg)
        # self.msg_frame_id = msg.header.frame_id
        self.msg_frame_id = "odom"
        self.msg_time_stamp = msg.header.stamp

        if not self.scnn_model:
            self.input_features = [
                field.name for field in msg.fields if field.name not in ["x", "y", "z","rgb"]]

            # Hacky way to deal with colour
            if "red" in self.input_features:
                self.input_colour_idx = [self.input_features.index(
                    "red"), self.input_features.index("green"), self.input_features.index("blue")]
                self.has_colour = True
            else:
                self.has_colour = False

        # Trigger to enable main cycle on next loop
        self.received_new_data_ = True

    def main_cb(self, event=None) -> ModuleNotFoundError:
        """Main callback to call all the process to enable the predictions data"""

        # Should we process?
        if not self.received_new_data_ or len(self.data_) < 1:
            return

        # Need the input_features so the model can be initialized
        if not self.scnn_model:
            self.scnn_model = ScnnFtmEnsemble(self.model_configs, self.device, self.input_features)

        # This is the most senseless and ugly conversion in a long while
        # Called a structured array in numpy and needs to be converted with a view
        start_time = time.perf_counter()

        X_coords = []
        feature_data = []
        # If we need to do the conversion
        start_0 = time.perf_counter()
        data_org = copy.deepcopy(self.data_)

        dtypes_coord = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ]
        X_coords = data_org[["x", "y", "z"]].astype(dtypes_coord).view(
            "<f4").reshape(data_org.shape[0], len(dtypes_coord))

        dtypes_feature = [(f_i, "<f4") for f_i in self.input_features]
        feature_data = data_org[self.input_features].astype(dtypes_feature).view(
            "<f4").reshape(data_org.shape[0], len(dtypes_feature))

        # Deal
        if self.has_colour:
            feature_data[:, self.input_colour_idx] = feature_data[:, self.input_colour_idx]/ 255.0 

        start_1 = time.perf_counter()

        # Prediction
        prediction_arr, _ = self.scnn_model.predict(
            X_coords=X_coords, X_features=feature_data, voxel_size=0.1)
        start_2 = time.perf_counter()

        #
        data_out_type = []
        use_rfb_cloud = True
        if use_rfb_cloud:
            data_out_type = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("label", "<f4"),
                    ("r", np.uint8),
                    ("g", np.uint8),
                    ("b", np.uint8),
                ])
        else:
            data_out_type = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("label", "<f4"),
                ]
            )

        # There must be a more efficient way of doing this
        pred_arr = np.empty(len(X_coords), dtype=data_out_type)
        pred_arr["x"] = X_coords[:, 0].view("f4").reshape(X_coords.shape[0])
        pred_arr["y"] = X_coords[:, 1].view("f4").reshape(X_coords.shape[0])
        pred_arr["z"] = X_coords[:, 2].view("f4").reshape(X_coords.shape[0])

        pred_arr["label"] = prediction_arr.astype("f4")

        colours = []
        if use_rfb_cloud:
            cmap = plt.cm.get_cmap('coolwarm_r')
            # cmap = plt.cm.get_cmap('bwr_r')
            colours = (cmap(prediction_arr)[:, 0:3]*255).astype(np.uint)
            # colours= np.vstack(
            #     [(255.0 * np.array(cmap(prob)[0:3])).astype(np.uint8) for prob in prediction_arr])
            pred_arr["r"] = colours[:, 0]
            pred_arr["g"] = colours[:, 1]
            pred_arr["b"] = colours[:, 2]

        start_3 = time.perf_counter()
        self.publish_predictions(pred_arr, use_rfb_cloud)
        start_4 = time.perf_counter()

        # Counters for time
        print(f"Elapsed time data conversion:  {  start_0 - start_time}")
        print(f"Elapsed time  ohmcloud_conversion:  {  start_1 - start_0}")
        print(f"Elapsed time  prediction:  {  start_2 - start_1}")
        print(f"Elapsed time  prediction conversion :  {  start_3 - start_2}")
        print(f"Elapsed time  publishing:  {  start_4 - start_3}")

        received_new_data = False
        self.data_ = []

    def publish_predictions(self, pred_arr, has_rgb):

        if has_rgb:
            pred_arr = rnp.point_cloud2.merge_rgb_fields(pred_arr)
        msg = rnp.msgify(PointCloud2, pred_arr)
        msg.header.frame_id = self.msg_frame_id
        msg.header.stamp = self.msg_time_stamp
        self.te_pub.publish(msg)
        print("Publishing prediction")


def main(args=None):
    rospy.init_node("ndt_te_estimator")

    node = ScnnFtmRosNode()

    rospy.Timer(rospy.Duration(0.1), node.main_cb)

    rospy.spin()


if __name__ == "__main__":
    main()
