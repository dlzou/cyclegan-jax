import argparse
import train
import predict
import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CycleGAN", description="CycleGAN")
    parser.add_argument("--train", help="Do training", action="store_true")
    parser.add_argument(
        "--predict", nargs=1, help="Do prediction starting from set A|B"
    )
    parser.add_argument(
        "-d",
        "--data-path",
        help="File path for training dataset or prediction file",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-path",
        help="File path for model checkpoints and outputs",
        required=True,
    )
    parser.add_argument(
        "--playground", help="Argument for testing", action="store_true"
    )

    arg = parser.parse_args()
    if arg.train:
        logger.info("Starting training!")
        model_opts, dataset_opts = train.get_default_opts(arg.data_path, arg.model_path)
        train.train(model_opts, dataset_opts)

    elif arg.predict:
        logger.info("Predicting file: {}".format(arg.data_path))
        model_opts, _ = train.get_default_opts(arg.data_path, arg.model_path)
        predict.predict(model_opts, arg.predict[0])

    elif arg.playground:
        logger.info("test")
        pass

    else:
        print("No flag specified")
        print("Use --train for training")
        print("Use --predict <path to file> for prediction")
