import os 
import numpy as np
import jax.numpy as jnp
from img_utils import img_to_array, array_to_img

def get_files_from_dir(dir_path: str):
	paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
	return paths

def path_to_array(path: str) -> jnp.ndarray:
	return jnp.expand_dims(img_to_array(path), axis=0) # 256 256 3 

def get_data_map(path):
	a = path[0]
	b = path[1]
	return {
		'A': path_to_array(a),
		'B': path_to_array(b), 
		'A_label': [a],
		'B_label': [b],
	}

def create_dataset():
	# Training: 
	trainA = get_files_from_dir('horse2zebra/trainA')
	trainB = get_files_from_dir('horse2zebra/trainB')
	# Zip train a and train b into tuple 
	train = list(zip(trainA, trainB))
	training_data = (get_data_map(t) for t in train)

	# Testing:
	testA = get_files_from_dir('horse2zebra/testA')
	testB = get_files_from_dir('horse2zebra/testB')
	# Zip test a and test b into tuple
	test = list(zip(testA, testB))
	testing_data = (get_data_map(t) for t in test)

	return training_data, testing_data
