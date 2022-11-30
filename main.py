import argparse
import config 
import data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog="CycleGAN", description="CycleGAN")
	parser.add_argument("--train", help="Flag for training", action='store_true')
	parser.add_argument("--predict", help="Flag for prediction, requires input path")
	arg = parser.parse_args()
	if arg.train:
		ds  = data.create_dataset()
		for i, v in enumerate(ds):
			# print("A IS!!!", v['A'])
			

			# print("B IS!!!", v['B'])
		
			breakpoint()
			break
		# train.train_and_evaluate()
		
	elif arg.predict:
		print("Predicting file: {}".format(arg.predict))


	else: 
		print("No flag specified")
		print("Use --train for training")
		print("Use --predict <path to file> for prediction")
	


	

