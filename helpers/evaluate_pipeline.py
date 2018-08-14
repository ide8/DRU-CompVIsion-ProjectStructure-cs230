"""Train pipeline function"""
import logging
import os

import tensorflow as tf

import importlib

from .utils import Params
from .utils import set_logger


def epipeline(args):
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    input_fn = importlib.import_module('models.{}.input_fn'.format(args.model)).input_fn
    model_fn = importlib.import_module('models.{}.model_fn'.format(args.model)).model_fn
    evaluate = importlib.import_module('models.{}.evaluation'.format(args.model)).evaluate

    job_dir = 'experiments/{}/{}/{}'.format(args.model, args.experiment, args.job_name)

    # Load the parameters
    json_path = os.path.join(job_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(job_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test_signs")

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    test_labels = [int(f.split('/')[-1][0]) for f in test_filenames]

    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, job_dir, params, args.restore_from)
