import argparse

import pandas as pd
import wandb

try:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install pytorch_lightning`."
    )

import yaml
from sklearn.metrics import classification_report, matthews_corrcoef

from scnn_tm.utils.setup_training import *


def train_cv(params):
    # print(params)
    
    model_files = {}
    for cv_nbr in range(params.max_cv):
        params.current_fold = cv_nbr
        params.cv_nbr = cv_nbr
        

        
        model_file = train_model(params)
        model_files[str(cv_nbr)] = {
            "y_pred":model_file["y_pred"],
            "y_target":model_file["y_target"],
            "mcc_score":model_file["mcc_score"],
            "f1_score": model_file["f1_score"]
        }
        
            
    if params.use_cv: 
        evaluate_cv_experiment(params, model_files)


def train_model(params):
    
    # Load models
    model_files = setup_model(params)

    # Load data set = load_data_set()
    model_files["data_loader"] = setup_data_factory_scnn(params, scaler=model_files["scaler"])

    # Load PL modules
    pl_module = setup_pl_module(params, model_files)

    #
    wandb_logger = setup_wandb_logger(params)
    
    # Scene based skip
    # if args.training_strategy_key == "cv_scene" and args.use_cv_scene_skip:
    #         if args.data_sets[current_fold].name in utils.data_set_name_without_nte():
    #             continue

    # Setup trainer
    trainer = Trainer(
        max_epochs=params.max_epochs,
        accelerator=params.accelerator,
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

    if params.save_model:
        simple_model_save(params, model_files)
    
    if params.evaluate_test_set:
        y_pred, y_target = evaluate_cv_run(params, model_files)
        model_files["y_pred"] = y_pred
        model_files["y_target"] = y_target
        mcc_score = matthews_corrcoef(y_pred=y_pred, y_true=y_target)
        model_files["mcc_score"]  = mcc_score
        wandb_logger.log_metrics({"test_mcc": mcc_score})
        
        
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

# TODO: Check that we can chnage/modify the values we care about with a sweep
CONFIG_FILE =  "/nve_ws/src/scnn_tm/config/default_config.yaml"
parser = argparse.ArgumentParser()


if __name__ == "__main__":
    # This loads the yaml file and maintains them in global scope. This allows wandb to change them
    # (CHECK): wand can change the files and this is
    # TODO: This is a dangerous parsing operation that wont allow much flexability
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
        for key, value in config.items():
            parser.add_argument(f"--{key}", type=type(value), dest=key, default=value)

    # Parse the command line arguments
    args = parser.parse_args()
    set_cv_exp_dir(args)

    train_cv(args)
