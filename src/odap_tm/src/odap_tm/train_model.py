# Copyright (c) 2023
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

import argparse

import pandas as pd
import wandb
import yaml
from sklearn.metrics import classification_report, matthews_corrcoef

from odap_tm.models.io_model_parsing import setup_model, simple_model_save
from odap_tm.models.io_plm_parsing import setup_pl_module

try:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install pytorch_lightning`."
    )


from odap_tm.setup_training import (
    evaluate_cv_experiment,
    evaluate_cv_run,
    set_cv_exp_dir,
    setup_data_factory,
    setup_wandb_logger,
)

# TODO: The model configs need the loss function for fine tuning

def train_cv(params):
    # print(params)
    
    model_files = {}
    for cv_nbr in range(params.max_cv):
        params.current_fold = cv_nbr
        params.cv_nbr = cv_nbr
        
        model_file = train_model(params)
        
        if params.evaluate_test_set:
            model_files[str(cv_nbr)] = {
            "y_pred":model_file["y_pred"],
            "y_target":model_file["y_target"],
            "mcc_score":model_file["mcc_score"],
            "f1_score": model_file["f1_score"]
            }
        
        # Jump out of the loop after one iteration
        if not params.use_cv:
            break

            
    if params.use_cv: 
        evaluate_cv_experiment(params, model_files)
        


def train_model(params):
    # Load models
    model_files = setup_model(params)

    # Load data set = load_data_set()
    model_files["data_loader"], model_files["scaler"] = setup_data_factory(params, scaler = model_files["scaler"])
    model_files["train_data_set"] = model_files["data_loader"].train_data_set
    model_files["val_data_set"] = model_files["data_loader"].val_data_set
    model_files["test_data_set"] = model_files["data_loader"].test_data_set
    model_files["params"] = params

    # Load PL modules
    pl_module = setup_pl_module(params, model_files, params.model_name)

    # Setup the logger if necessary
    wandb_logger = setup_wandb_logger(params)

    # Setup trainer
    trainer = Trainer(
        max_epochs=params.max_epochs,
        accelerator=params.accelerator,
        detect_anomaly = False,
        devices=1,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(
                monitor=f"val_loss",
                patience=params.es_patience,
                mode="min",
            )
        ],
    )
    trainer.fit(pl_module)

    # Setup the outdir after loading the models in case pre-trained models are used
    if params.cv_exp_dir is None:
        set_cv_exp_dir(params)

    if params.save_model:
        simple_model_save(params.cv_exp_dir, params, model_files)
    
    if params.evaluate_test_set:
        y_pred, y_target = evaluate_cv_run(params, model_files)
        
        mcc_score = matthews_corrcoef(y_pred=y_pred, y_true=y_target)
        model_files["mcc_score"]  = mcc_score
        wandb_logger.log_metrics({"test_mcc": mcc_score})
        
        # Why are we storing this here?
        model_files["y_pred"] = y_pred
        model_files["y_target"] = y_target
        report = classification_report(y_true=y_target, y_pred=y_pred, output_dict=True)
        model_files["f1_score"] = report["macro avg"]["f1-score"]
        # model_files["recall"] = report["macro avg"]["recall"]
        params.result_file = params.cv_exp_dir / "exp_results.txt"
        f = open(params.result_file, "a+")
        f.write(f"\n \n Report for fold n {params.cv_nbr} \n")
        f.write(f"MCC-Score = {mcc_score} \n \n")
        f.write(classification_report(y_true=y_target, y_pred=y_pred))
        f.close()
        
    wandb.finish()

    return model_files


###################################################
#        Main Loop and Arguments
###################################################

# TODO: Check that we can chnage/modify the values we care about with a sweepsrc/odap_tm/cfg/run_23_10_18_pu_training.s
# CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_04_23/inustrial_base.yaml"
# CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_04_23/sparse_forest_base.yaml"
# CONFIG_FILE = "/nve_ml_ws/src/nve_ml_config/offline_processing/2024_04_23/dense_forest_base.yaml"


parser = argparse.ArgumentParser()


if __name__ == "__main__":
    # This loads the yaml file and maintains them in global scope. This allows wandb to change them
    # (CHECK): wand can change the files and this is
    # TODO: This is a dangerous parsing operation that wont allow much flexibility
    CONFIG_FILE = "/nve_ws/src/odap_tm/config/default_config.yaml"
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
        for key, value in config.items():
            parser.add_argument(f"--{key}", type=type(value), dest=key, default=value)

    # Parse the command line arguments
    args = parser.parse_args()
    args.cv_exp_dir = None

    train_cv(args)
