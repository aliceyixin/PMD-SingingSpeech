# coding: utf-8
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import importlib
from utils.data_io_ss import prepare_datasets
import os
from speechbrain.utils.profiling import profile, schedule
import logging
from thop import profile, clever_format
import numpy as np
# Recipe begins!
if __name__ == "__main__":

    # parse command line args
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # load hparams
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create experiment directory
    sb.create_experiment_directory(
            experiment_directory=hparams['output_dir'],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
    )

    # json file preparation
    dataset_name = hparams['dataset']
    importlib.import_module(f'datasets.{dataset_name}.prepare_da_json').prepare(**hparams['prepare'])

    # Load parsed dataset
    datasets, label_encoder = prepare_datasets(hparams)
    train_dataset, valid_dataset, test_dataset = datasets
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

     # initialize model
    if 'model_class' in hparams:
        model_class = hparams['model_class']
        SBModel = importlib.import_module(f'models.{model_class}.model').SBModel
        model = SBModel(
            label_encoder=label_encoder,
            modules=hparams['modules'],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams['checkpointer'],
        )


    # fit the model
    model.fit(
        hparams['epoch_counter'],
        train_dataset,
        valid_dataset,
        train_loader_kwargs=hparams['train_dataloader_opts'],
        valid_loader_kwargs=hparams['valid_dataloader_opts'],
    )

    # test the model
    model.evaluate(
        test_dataset,
        max_key=hparams['max_key'],
        min_key=hparams['min_key'],
        test_loader_kwargs=hparams['test_dataloader_opts'],
    )

    # logger = logging.getLogger(__name__)
    # if hparams["compute_macs"] :
    #     macs, params = clever_format([np.mean(model.macs), np.mean(model.params)], "%.3f")
    #     print("macs, params: ", macs, params)
    #     logger.info(macs)
    #     logger.info(params)