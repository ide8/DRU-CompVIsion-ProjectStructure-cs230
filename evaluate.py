"""Evaluate the model"""

import argparse

from helpers.evaluate_pipeline import epipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None,
                    help="Model in models directory")
parser.add_argument('--experiment', default=None,
                    help="Experiment in experiments directory wrt model")
parser.add_argument('--job_name', default='model',
                    help="Job name")
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()

    # Start evaluation pipeline
    epipeline(args)
