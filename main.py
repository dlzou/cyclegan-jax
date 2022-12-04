import argparse
import train
import predict
import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CycleGAN", description="CycleGAN")
    parser.add_argument("--train", help="Flag for training", action="store_true")
    parser.add_argument("--predict", help="Flag for prediction, requires input path")
    parser.add_argument("--path", help="File path for training dataset", required=True)
    parser.add_argument(
        "--playground", help="Argument for testing", action="store_true"
    )

    arg = parser.parse_args()
    if arg.train:
        logger.info("Starting training!")
        model_opts, dataset_opts = train.get_default_train_ops(arg.path)
        train.train(model_opts, dataset_opts)

    elif arg.predict:
        logger.info("Predicting file: {}".format(arg.predict))
        predict.predict(train.model_opts, arg.predict)

    elif arg.playground:
        logger.info("test")
        pass

    else:
        print("No flag specified")
        print("Use --train for training")
        print("Use --predict <path to file> for prediction")
