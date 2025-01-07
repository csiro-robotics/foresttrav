# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz

#!/usr/bin/env python   
from dataclasses import dataclass
from pathlib import Path

import rospy
import torch

try:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install pytorch_lightning`."
    )

import yaml
from joblib import dump
import datetime

from odap_tm.models.OnlineLearingModule import OnlineActiveLearingModule
from ros_scnn_tm.msg import OnlineDataSet, TriggerModelUpdate
from scnn_tm.utils import generate_feature_set_from_key, load_yamls_as_struct



class OnlineLearingParamsParser:
    def __init__(self, **entries) -> None:
        self.__dict__.update(entries)


class OnlineLearningServer:
    """An online trainer that trains model on the go"""

    def __init__(self):
        # Empty member variables used for initalizations
        self.model = None
        self.scaler = None
        self.new_training_files = []
        self.model_iteration_num = 0
        self.last_best_mcc_score = 0.0

        # Load parameters
        self.load_ros_params()

        self.online_learner = OnlineActiveLearingModule(self.learning_params)

        # Finaly invoke the timers?
        self.pub_model_update = rospy.Publisher(
            "~online_model_update_trigger", TriggerModelUpdate, queue_size=1
        )
        self.learning_service_cb = rospy.Subscriber(
            "~new_training_data", OnlineDataSet, self.new_data_cb
        )
        self.main_cb_timer = rospy.Timer(rospy.Duration(1.0), self.main_cb)
        
        # List of the current config files  
        self.model_config_files = []

    def process_incoming_data(self):
        # Recude learning rate if necessary
        sorted_file_names = sorted(self.new_training_files)
        self.online_learner.update_online_dataset(sorted_file_names)
        self.new_training_files = []
        

    def main_cb(self, event=None):
        
        # Named?
        rospy.logdebug_throttle(10.0, "Wanting to trainning a new model")
        if not self.new_training_files :
            return

        # Process new training data
        self.process_incoming_data()
        
        # Train new model
        # The models are timestamped anyway, so only makes a difference once we train more than 1 model
        # TODO(fab): Parallel train n-models   
        # self.online_learner.parallel_train_models([model_num_1, model_num_2, model_num_3])
        
        # TODO(fab): Make the bottom an async task? Or wait for the model to finish. 
        rospy.loginfo("Trainning a new model")
        model_file, mcc_score = self.online_learner.train_new_model(0)
        
        # The model can fail to train if there is not enough data to train.
        if model_file is None or type(mcc_score) is str:
            rospy.logwarn(f"Failed to train online model: {mcc_score}")
            return None
        
        # Save the model
        rospy.loginfo(f"Trainning a new model #{self.model_iteration_num} with MCC score: {mcc_score}")
        model_out_dir, iter_num, config_file = self.save_model(
                model=model_file["model"],
                scaler=model_file["scaler"],
                feature_set_key=model_file["params"].feature_set_key,
                model_iter_number=self.model_iteration_num,
            )
        
  
        update_model = self.learning_params.cont_learning
        publish_model = self.publish_model_update_info
        
        # Case 1: MCC is not available - Nothing needs to be done
        if mcc_score is None or not self.use_best_model:
            update_model = self.learning_params.cont_learning
            publish_model = self.publish_model_update_info
            
        # Case 2: Online eval possible and only the 'best' model is considere
        elif self.use_best_model:
            
            if mcc_score > self.last_best_mcc_score:
                self.last_best_mcc_score = mcc_score 
                update_model =  self.learning_params.cont_learning and True
                publish_model =  self.publish_model_update_info and True
        
            else:
                update_model =  self.learning_params.cont_learning and False
                publish_model =  self.publish_model_update_info and False
                
        else:
            msg = "Invalid update step occured"
            raise ValueError(msg)
           
        
        # Updatate model if CL
        if update_model:
                self.online_learner.base_models[0] = model_file
            
        # Publish model update
        if publish_model:
            msg = TriggerModelUpdate()
            msg.cv_number = 10
            msg.cv_start = 0
            msg.file_path = str(model_out_dir)
            self.pub_model_update.publish(msg)
            
        self.model_iteration_num = +1

    def load_ros_params(self):
        
        # Prints the information finetune and debug
        self.debug_print = rospy.get_param("~debub_print", False)
        
        # R
        self.model_out_dir = Path(rospy.get_param("~model_out_dir", None))
        
        # Adds the datetime in format YYYY_MM_DDD_HH_MM_SS to the directory to make an unique experiment id.
        self.use_daytime_stamp = rospy.get_param("~use_datestamp", True)
        if self.use_daytime_stamp:
            now = datetime.datetime.now()
            self.model_out_dir = self.model_out_dir / now.strftime("%Y_%m_%d_%H_%M_%S")
        
        self.learning_params_file = Path(rospy.get_param("~learning_param_file", None))

        # Broadcast the update info for the network
        self.publish_model_update_info = rospy.get_param("~publish_model_update_info", True)

        # Load a previous model
        self.pretrained_model_file = Path(rospy.get_param("~pretrained_model_file", ""))

        # Flag to train from scratch
        self.use_best_model = rospy.get_param("~use_best_model", True)
         
            
        self.learning_params = load_yamls_as_struct(self.learning_params_file)
        feature_set = generate_feature_set_from_key(
            self.learning_params.feature_set_key
        )

        # Set new attributes
        self.learning_params.set_new_attribute("feature_set", feature_set)
        self.learning_params.set_new_attribute("data_set_file", "")

    
    # TODO: Use the default functions and avoid this abomination
    def save_model(
        self, model, scaler, feature_set_key=None, model_iter_number: int = 0
    ):
        time_str = rostime_to_file_string(rospy.Time.now())
        now = datetime.datetime.now()
        model_out_dir = Path(self.model_out_dir) / now.strftime("%Y_%m_%d_%H_%M_%S")
        model_out_dir.mkdir(parents=True, exist_ok=False)

        # Save model
        model_pl_file = (
            model_out_dir / f"{model.__class__.__name__}_cv_{model_iter_number}.pl"
        )
        torch.save(model.state_dict(), model_pl_file)

        # Save the data scaler
        scaler_file = model_out_dir / f"scaler_cv_{model_iter_number}.joblib"
        dump(scaler, scaler_file)

        # Save the config file

        config_file = model_out_dir / f"model_config_cv_{model_iter_number}.yaml"
        config_file.touch()
        with open(config_file, "w") as file_descriptor:
            yaml_params = {}
            yaml_params["model_name"] = model.__class__.__name__
            yaml_params["model_nfeature_enc_ch_out"] = model.ENCODER_CH_OUT[0]
            yaml_params[
                "model_skip_connection_key"
            ] = f"s{model.SKIP_CONNECTION.count(1)}"
            yaml_params["model_dropout"] = 0.05
            yaml_params["feature_set_key"] = feature_set_key

            # We make the assumption that the model is in the same dir
            yaml_params["torch_model_path"] = str(model_pl_file)
            yaml_params["scaler_path"] = str(scaler_file)
            yaml_params["timestamp"] = time_str
            yaml.safe_dump(yaml_params, file_descriptor)

        return model_out_dir, model_iter_number, config_file

    def train_srv_cb(self, request):
        new_training_file = Path(request.file_path)

        response = TriggerModelUpdate()
        if new_training_file.is_file():
            self.new_training_files = Path(request.file_path)
            self.last_data_stamp = request.stamp
            response.success = True
            response.message = ""
        else:
            response.success = False
            response.message = f"Could not find file {str(self.new_training_files)}"
        return response

    def new_data_cb(self, msg):
        """Callback to assess if a new data is received to trigger another training instance"""
        new_training_file = Path(msg.file_path)
        if not new_training_file.exists():
            rospy.WARN(f"fCould not find online data set: {str(new_training_file)}")

        self.new_training_files.append(new_training_file)
        self.last_data_stamp = msg.stamp


def rostime_to_string(rostime):
    return f"{rostime.secs}.{rostime.nsecs:09d}"


def rostime_to_file_string(rostime):
    return f"{rostime.secs}_{rostime.nsecs:09d}"


def main(args=None):
    rospy.init_node("scnn_server")

    node = OnlineLearningServer()

    rospy.spin()


if __name__ == "__main__":
    main()
#
