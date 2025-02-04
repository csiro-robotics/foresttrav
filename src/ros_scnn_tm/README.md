# ROS Sparse Convolutional Neural Network for Traversability Mapping
Contains the ros node to generate online te estimation. A default launch and config file. This models uses "ensembles". The models are loaded based on the model_dir params, and CV_MIN and CV_MAX define the range of which models should be loaded , i.e. 2-5. ( easy to switching models). 

## QUICkSTART
To launch a node with the default values, see `ros_snn_tm_node.launch`. The default params are defined in the [config_file](config/default_ros_scnn_node.yaml)

` roslaunch ros_scnn_tm ros_scnn_tm_node.launch`

Models can be changed online using the update_model_dir service. 

## Online learning module:
Is under development and changing.

