"""Train pipeline function"""
import logging
import os

import tensorflow as tf

import importlib

from .utils import Params
from .utils import set_logger

def tpipeline(args):
    # Set the random seed for the whole graph for reproducible experiments
    tf.set_random_seed(230)

    input_fn = importlib.import_module('models.{}.input_fn'.format(args.model)).input_fn
    model_fn = importlib.import_module('models.{}.model_fn'.format(args.model)).model_fn
    train_and_evaluate = importlib.import_module('models.{}.training'.format(args.model)).train_and_evaluate

    experiment_dir = 'experiments/{}/{}'.format(args.model, args.experiment)

    json_path = os.path.join(experiment_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Create a new folder in parent_dir with unique_name "job_name"
    # If folder exists try to get params from there
    experiment_dir = os.path.join(experiment_dir, args.job_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    elif os.path.exists(os.path.join(experiment_dir, 'params.json')):
        params = Params(os.path.join(experiment_dir, 'params.json'))
    else:
        # Write parameters in json file
        json_path = os.path.join(experiment_dir, 'params.json')
        params.save(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwriting
    experiment_dir_has_best_weights = os.path.isdir(os.path.join(experiment_dir, "best_weights"))
    overwriting = experiment_dir_has_best_weights and args.restore_from is None
    assert not overwriting, "Weights found in experiment_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(experiment_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train_signs")
    dev_data_dir = os.path.join(data_dir, "dev_signs")

    # Get the file names from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                       if f.endswith('.jpg')]
    eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                      if f.endswith('.jpg')]

    # Labels will be between 0 and 5 included (6 classes in total)
    train_labels = [int(f.split('/')[-1][0]) for f in train_filenames]
    eval_labels = [int(f.split('/')[-1][0]) for f in eval_filenames]

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, experiment_dir, params, args.restore_from)
