import argparse
import config
import data
import train
from logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CycleGAN", description="CycleGAN")
    parser.add_argument("--train", help="Flag for training", action="store_true")
    parser.add_argument("--predict", help="Flag for prediction, requires input path")
    parser.add_argument("--playground", help="Argument for testing", action="store_true")

    arg = parser.parse_args()
    if arg.train:
        logger.info("Starting training!")
        train.train(train.model_opts, train.dataset_opts)

    elif arg.predict:
        print("Predicting file: {}".format(arg.predict))

    elif arg.playground: 
        pass

    else:
        print("No flag specified")
        print("Use --train for training")
        print("Use --predict <path to file> for prediction")
