# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

from copy import deepcopy
from pathlib import Path
from joblib import load
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

try:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install pytorch_lightning`."
    )
from odap_tm.setup_training import evaluate_cv_model
from odap_tm.models.io_model_parsing import setup_model
from odap_tm.models.io_plm_parsing import setup_pl_module
from scnn_tm.models.ForestTravDataSet import ForestTravDataReader, ForestTravDataSet
from scnn_tm.models.ForestTravDataSetFactory import (
    initalize_and_fit_scaler,
    patchify_data_set,
    scale_data,
    DataPatchAugmenter,
)
from scnn_tm.models.ForestTravGraph import ForestTravGraph
from scnn_tm.models.OnlineFotrestTravDataLoader import OnlineDataSetLoader
from scnn_tm.utils import generate_feature_set_from_key

# TODO: Setup logger for the class
""" Online active learning module allows to train models on the fly whilst deployed on a robot

    Online learning strategies:
    We differentiate for four different strategies: 
        1. new_model_each_iter: 
            - Creates a new model for every instance of training and discards the previous model.
                 base_models" (None or list): Empty list
                ft_last_trained_model (bool): False
            
        2. from_base_model_each_iter:  For each iteration  the model is finetuned from a initialised base model. The trained models are discarded afterwards
                base_models (list): List of base model configs that need to be loaded
                ft_last_trained_model (bool): False
            
    TODO: Learning rate reduction for fine-tune or different iterations
        3. cl_w_new_model:  Creates a new model at the initialisation, fine tunes each consequetive iteration
                base_models" (None or list): Empty list
                ft_last_trained_model (bool): False
            
        4. cl_w_base_model: Loades a base model and learns contiouly with that model over each iteration. Contious ft model for each iteration 
            base_models (list): List of base model configs that need to be loaded
            ft_last_trained_model (bool): True 

    Data set strategies (data_set_strategy_key):
        - {online}:             Only use the latest only data. Discards all the rest
        - {base_train_data}:    Loads base
    

    Online validation strategy:
        - {online_val}:     Validates on the validation test set, split off from training set
        - {test_data}:      Validates on defined test data set, can either be a csv test set or a data graph at time t

"""

# TODO: Setup a parallel learning framework?
# TODO: Setup the learning/training framwork so the interface beocmes only train model X


class OnlineActiveLearingModule:
    """An online trainer that trains model on the go"""

    def __init__(self, params):
        # Class members
        self.ft_models_ = []

        # Empty member variables used for initalizations
        self.model_files = []
        self.scaler = None
        self.new_training_files = []
        self.model_iteration_num = 0
        self.mcc_model_scores = {}

        # Local parameters
        self.params = params

        # Set the feature set if not already set
        self.feature_set = None
        if not params.use_pretrained_model:
            self.feature_set = generate_feature_set_from_key(
                params.feature_set_key
            )  # All models need to use the same feature set

        # These are global params
        self.voxel_size = params.voxel_size
        self.nvoxel_leaf = params.nvoxel_leaf
        self.min_patch_sample_nbr = params.min_patch_sample_nbr
        self.patch_width = params.patch_width
        self.min_dist = params.min_pose_dist
        
        # Default initialistaion (Hack)
        if hasattr(params, "min_num_training_samples"): 
            self.min_number_training_samples = params.min_num_training_samples
        else:
            self.min_number_training_samples = 10
            
        # Setup the base models and data sets that persists oves the learing iterations
        self.setup_base_models(params)

        self.setup_base_data_sets()

        # Online Graph Data set
        self.online_train_data = ForestTravGraph(
            voxel_size=self.voxel_size,
            patch_width=self.patch_width,
            min_dist_r=self.min_dist,
            use_unlabeled_data=params.use_unlabelled_data,
        )

        # Check that all the configurations are valid
        self.check_valid_config()

    def setup_base_models(self, params):
        """Helper function to setup the model"""

        # TODO: This may contain n models
        self.base_models = []
        self.base_train_data = []

        # Load all the necessary files to finetune
        for model_num in range(params.max_cv):
            # The deep copy is to avoid overloading the model params to other models/files
            model_params = deepcopy(params)
            self.base_models.append(setup_model(model_params))
            self.base_models[-1]["cv"] = model_num
            self.base_models[-1]["params"] = deepcopy(model_params)
            self.base_models[-1]["iter"] = 0  # Finetune iterations

            if self.feature_set is None:
                self.feature_set = model_params.feature_set

        # Check if there is a default scaler that should be used
        if params.use_default_scaler and not params.use_pretrained_model:
            self.default_scaler = load(params.default_scaler_path)
            for model_files in self.base_models:
                    model_files["scaler"] =  self.default_scaler
                    

    def setup_learning(self, model_files):
        params = model_files["params"]
        plm = setup_pl_module(
            model_files=model_files, params=params, plm_tag=params.model_name
        )

        # Setup the trainne
        trainer = Trainer(
            max_epochs=self.params.max_epochs,
            accelerator=self.params.accelerator,
            devices=1,
            check_val_every_n_epoch=self.params.check_val_every_n_epoch,
            callbacks=[
                EarlyStopping(
                    monitor="train_loss",
                    patience=self.params.es_patience,
                    mode="min",
                )
            ],
        )

        return trainer, plm

    def evaluate_model(self, model_files):

        y_pred, y_target = evaluate_cv_model(
            model=model_files["model"], test_data_set=model_files["test_data_set"]
        )

        return matthews_corrcoef(y_true=y_target, y_pred=y_pred)

    def setup_base_data_sets(self):

        # If we do not have self
        if self.params.base_train_data_sets:

            # Setup the train data we want to have
            base_train_data_set_csv = [
                file_path
                for file_path in self.params.base_train_data_sets
                if "csv" in str(file_path)
            ]

            # Load the raw data set
            self.base_train_data = []
            if base_train_data_set_csv:
                base_train_data_raw = ForestTravDataReader(
                    data_sets_files=base_train_data_set_csv,
                    feature_set=self.feature_set,
                ).raw_data_set

                self.base_train_data = patchify_data_set(
                    raw_data=base_train_data_raw,
                    patch_strategy="grid",
                    voxel_size=self.voxel_size,
                    nvoxel_leaf=self.nvoxel_leaf,
                    min_patch_sample_nbr=self.min_patch_sample_nbr,
                )

        if self.params.base_test_data_sets:
            # Setup the test data so we can evaluate the models
            base_test_data_set_csv = [
                file_path
                for file_path in self.params.base_test_data_sets
                if "csv" in str(file_path)
            ]

            if base_test_data_set_csv:
                self.test_data_set = ForestTravDataReader(
                    data_sets_files=base_test_data_set_csv,
                    feature_set=self.feature_set,
                ).raw_data_set

    def update_online_dataset(self, new_data_sets: list):
        """Updates the graph with newly received data.

        Args:
            new_data_sets (list): A list of files (hdf5) which contain the new data.

        """

        data_loader = OnlineDataSetLoader(
            target_feature_set=self.feature_set,
            voxel_size=self.voxel_size,
        )
        # Sort again to be
        new_data_sets.sort()
        for new_data_set_file in reversed(new_data_sets):
            new_data_batch = data_loader.load_online_data_set_filtered(
                Path(new_data_set_file)
            )
            self.online_train_data.add_new_data(new_data_batch=new_data_batch)

    def train_new_model(self, model_nbr):

        # TODO: Decide what to do with the copy
        model_files_i = deepcopy(self.base_models[model_nbr])

        # Generate the k sets needed to train
        if not self.prepare_data_for_model(model_files_i):
            return None,None
        
        # Exit if there is not enoug data to learn on
        if len(model_files_i['train_data_set']) < self.min_number_training_samples:
            return None, None
            
        # Start training the model
        trainer, plm = self.setup_learning(model_files=model_files_i)

        trainer.fit(plm)
        
        mcc_score = None
        if len(self.test_data_set) > 0:
            mcc_score = self.evaluate_model(model_files_i)

        # TODO: This will overwrite files if not carefull
        model_files_i["params"].cv_nbr = model_nbr

        # Remove non-essntial information
        model_files_i["train_data_set"] = None
        model_files_i["val_data_set"] = None

        return model_files_i, mcc_score


    def prepare_data_for_model(self, model_files):
        """Preparses the data for the model training. Note that the data needs to be copyied to      avoid mutating it for other training instances."""

        # Take the base data sets and the online data set, and split it into train/val test set
        oline_train_data = self.online_train_data.get_patch_data_set_copy()

        # Get the
        raw_train_data = deepcopy(self.base_train_data) + oline_train_data
        
        if len(raw_train_data) < self.min_number_training_samples:
            return False

        # Check the data has the desired format, front and back will get both data sets if available
        for f_data in [raw_train_data[0], raw_train_data[-1]]:
            if f_data["features"].shape[1] != len(self.feature_set):
                raise ValueError("Feature map has wrong dimensions")

        # Note, these are unscaled at this point
        train_set = []
        val_set = []
        if self.params.train_ratio < 0 or self.params.train_ratio >= 1.0:
            train_set = raw_train_data
        else:
            train_set, val_set = train_test_split(
                raw_train_data,
                train_size=self.params.train_ratio,
                random_state=model_files["params"].random_seed,
                shuffle=True,
            )

        # Split the data into train/val data set
        # TODO: Check if we need a deep copy here
        model_files["train_data_set"] = train_set
        model_files["val_data_set"] = val_set

        # Copy the test data ss
        model_files["test_data_set"] = deepcopy(self.test_data_set)

        # Setup scaler if necessary and scale the data
        self.fit_and_scale_data(model_files=model_files)

        # Setup the DataLoaders
        self.generate_forest_trav_data_sets(model_files=model_files)
        
        # The process was successfull
        return True

    def generate_forest_trav_data_sets(self, model_files):
        """Generates the DataSet used by pytorch

        Args:
            train_data (_type_): _description_
            val_data (_type_): _description_
            test_data (_type_): _description_
        """
        model_files["train_data_set"] = ForestTravDataSet(
            data_set=model_files["train_data_set"],
            voxel_size=self.voxel_size,
            data_augmentor=DataPatchAugmenter(
                voxel_size=self.params.voxel_size,
                noise_chance=self.params.data_augmenter_noise_chance,
                noise_mean=self.params.data_augmenter_noise_mean,
                noise_std=self.params.data_augmenter_noise_std,
                sample_pruning_chance=self.params.data_augmenter_sample_pruning_chance,
                rotation_chance=self.params.data_augmenter_batch_rotation_chance,
                translation_chance=self.params.data_augmenter_batch_translation_chance,
                n_voxel_displacement=self.params.data_augmenter_batch_nvoxel_displacement,
                mirror_chance=self.params.data_augmenter_mirror_chance,
            ),
            use_data_augmentation=model_files["params"].use_data_augmentation,
        )

        model_files["val_data_set"] = ForestTravDataSet(
            data_set=model_files["val_data_set"],
            voxel_size=self.params.voxel_size,
            data_augmentor=None,
            use_data_augmentation=False,
        )

        model_files["test_data_set"] = ForestTravDataSet(
            data_set=model_files["test_data_set"],
            voxel_size=self.params.voxel_size,
            data_augmentor=None,
            use_data_augmentation=False,
        )

    def fit_and_scale_data(self, model_files):

        # Check if the attribut is set, only for valid initalised scalers
        if not hasattr(model_files["scaler"], "n_features_in_"):
            model_files["scaler"] = initalize_and_fit_scaler(
                data_train=model_files["train_data_set"]
            )

        model_files["train_data_set"] = scale_data(
            model_files["scaler"], model_files["train_data_set"]
        )
        model_files["val_data_set"] = scale_data(
            model_files["scaler"], model_files["val_data_set"]
        )

        model_files["test_data_set"] = scale_data(
            model_files["scaler"], model_files["test_data_set"]
        )

    def check_valid_config(self) -> None:
        """Checks for valid configurations"""
        
        return True
        # There is only one option for usig unlabaled data
        # if self.params.use_unlabeled_data:

        #     # Needs to be THM model and have the TwoHead
        #     for model_files in self.base_models:
        #         if (
        #             model_files["params"].loss_function_tag != "TwoHeadProbLoss"
        #             or "THM" not in model_files["params"].model_name
        #         ):
        #             raise ValueError(
        #                 f"Invalid configuration used: Unlabeled data and without a propper loss funvtion '{model_files['params'].loss_function_tag}'"
        #             )
