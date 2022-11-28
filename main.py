import argparse
import config 
import pprint
import train

if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog="Cyclegan", description="CycleGAN")
	parser.add_argument("--train", help="Flag for training", action='store_true')
	parser.add_argument("--predict", help="Flag for prediction, requires input path")
	arg = parser.parse_args()
	if arg.train:
		print("Training with configuration: ")
		pprint.pprint(config.configurations)
		train.train_and_evaluate()
		
	elif arg.predict:
		print("Predicting file: {}".format(arg.predict))
	else: 
		print("No flag specified")
		print("Use --train for training")
		print("Use --predict <path to file> for prediction")
	


	

