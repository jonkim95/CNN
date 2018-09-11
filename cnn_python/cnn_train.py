import cnn
import cnn_data as data

import numpy as np
import pickle

BATCH_SIZE = 30
LEARNING_RATE = 0.01
NUM_EPOCHS = 3
FILENAME = "saved_params.p"

def main():
	data_dicts = data.get_data()
	num_channels = data.get_num_channels()
	img_height = data.get_image_height()
	img_width = data.get_image_width()

	print("Initializing parameters")
	kernel, conv_bias, fc_weights, fc_bias = cnn.init_params(zeros=False)
	for epoch in xrange(NUM_EPOCHS):
		np.random.shuffle(data_dicts)
		batches = [data_dicts[k:k + BATCH_SIZE] for k in xrange(0, len(data_dicts), BATCH_SIZE)]
		for batch_num, batch in enumerate(batches):
			print("Starting batch {} of epoch {}".format(batch_num, epoch))
			params, num_correct, net_loss = cnn.update_params(batch, kernel, conv_bias, fc_weights, fc_bias, LEARNING_RATE)
			avg_loss = net_loss / len(batch)
			print("Accuracy : {} / {}".format(num_correct, BATCH_SIZE))
			print("Mean loss : {}\n".format(avg_loss))
			kernel = params['kernel']
			conv_bias = params['conv_bias']
			fc_weights = params['fc_weights']
			fc_bias = params['fc_bias']
			
	final_params = [kernel, conv_bias, fc_weights, fc_bias]
	to_save = [final_params, num_correct, avg_loss]

	pickle.dump(to_save, open(FILENAME, "wb"))

if __name__ == "__main__":
	main()