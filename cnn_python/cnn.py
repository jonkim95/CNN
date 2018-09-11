import cnn
import cnn_data as data

import math
import numpy as np
import pdb
import pickle

KERNEL_SIZE = 3
KERNEL_STRIDE = 1
POOL_SIZE = 2
POOL_STRIDE = 2

CLIP_CONSTANT = np.float128(np.exp(-7))

ERROR_LOG = "error_log.p"

# General parameter matrix initialization function based on http://cs231n.github.io/neural-networks-2/#init
def init_matrix(h, w, d, zeros=False):
	if zeros: return np.float128(np.zeros((h, w, d)))
	return np.random.randn(h, w, d) * math.sqrt(2.0 / (h * w * d))

def init_2d_matrix(h, w):
	return np.random.randn(h, w) * math.sqrt(2.0 / (h * w))

def softmax(X):
    out = np.exp(X - max(X))
    return out/np.sum(out)

def clip(X):
	X[X == 0] += CLIP_CONSTANT
	return X

def cross_entropy(predictions, targets):
    num_examples = predictions.shape[0]
    predictions = clip(predictions)
    ce = - np.sum(np.sum(targets * np.log(predictions))) / num_examples
    if math.isnan(ce):
    	to_save = [predictions, targets]
    	pickle.dump(to_save, open(ERROR_LOG, "wb"))
    return ce

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

def convolve(img, kernel, bias):
	num_channels = data.get_num_channels()
	num_fit_h = (data.get_image_height() - KERNEL_SIZE) / KERNEL_STRIDE + 1 # Number of filters that "fit" in a column of the input image
	num_fit_w = (data.get_image_width() - KERNEL_SIZE) / KERNEL_STRIDE + 1 # Number of filters that "fit" in a row of the input
	conv_layer = np.zeros((num_fit_h, num_fit_w))
	for i in xrange(0, num_fit_h, KERNEL_STRIDE):
		for j in xrange(0, num_fit_w, KERNEL_STRIDE):
			conv_layer[i, j] = np.sum(img[i:i+KERNEL_SIZE, j:j+KERNEL_SIZE, :] * kernel) + bias
	#print(conv_layer)
	return conv_layer

def backward_convolve(d_conv_layer, orig_img, kernel):
	num_channels = data.get_num_channels()
	d_kernel = np.zeros(kernel.shape)
	for i in xrange(len(d_conv_layer)):
		for j in xrange(len(d_conv_layer)):
			d_kernel += orig_img[i:i + KERNEL_SIZE, j:j + KERNEL_SIZE] * d_conv_layer[i, j]
            # loss gradient of the input to the convolution operation (conv1 in the case of this network)
       		j += KERNEL_STRIDE
        i += KERNEL_STRIDE
	d_conv_bias = np.sum(d_conv_layer)
	return d_kernel, d_conv_bias
    
def maxpool(mat, size, stride):
	(h, w) = mat.shape
	pooled = np.zeros(((h - size) / stride + 1, (w - size) / stride +1))
	i = 0
	while i < h:
		j = 0
		while j < w:
			pooled[i / 2, j / 2] = np.max(mat[i:i + size, j:j + size])
			j += stride
		i += stride	
	return pooled

# Based on https://gist.github.com/Alescontrela/c57664c9cd0a9689789363e9820e5d01
def backward_maxpool(d_pooled_layer, orig_mat, size, stride):
    '''
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    '''
    (orig_h, orig_w) = orig_mat.shape
    d_conv_layer = np.zeros(orig_mat.shape)
    i = 0
    output_i = 0
    while i + size <= orig_h:
    	j = 0
    	output_j = 0
    	while j + size <= orig_w:
    		# obtain index of largest value in input for current window
        	(a, b) = nanargmax(orig_mat[i:i + size, j:j + size])
        	d_conv_layer[i + a, j + b] = d_pooled_layer[output_i, output_j]
	        j += stride
	        output_j += 1
        i += stride
        output_i += 1    
    return d_conv_layer

def learn_gradients(probs, label, layers, fc_weights, fc_bias, img, kernel):
	conv_layer = layers['conv']
	pooled_layer = layers['pooled']
	fc_layer = layers['fc']
	d_output = probs - label # derivative of loss with respect to the final dense layer output
	d_fc_weights = np.reshape(d_output, (10,1)).dot(fc_layer.T) # loss gradient of final dense layer weights
	d_fc_bias = np.reshape(d_output, fc_bias.shape) # loss gradient of final dense layer biases
	d_fc_layer = np.reshape(fc_weights.T.dot(d_output), fc_layer.shape) # loss gradient of fully connected layer
	d_fc_layer[fc_layer <= 0.0] = 0.0 # backpropagate through ReLU
	d_pooled_layer = np.reshape(d_fc_layer, pooled_layer.shape) # reshape fully connected into dimensions of pooling layer
	d_conv_layer = backward_maxpool(d_pooled_layer, conv_layer, POOL_SIZE, POOL_STRIDE) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
	d_conv_layer[conv_layer <= 0.0] = 0.0 # backpropagate through ReLU
	[d_kernel, d_conv_bias] = backward_convolve(d_conv_layer, img, kernel) # backpropagate previous gradient through first convolutional layer
	return [d_kernel, d_conv_bias, d_fc_weights, d_fc_bias]

def calc_fc_size(num_channels):
	num_fit_h = (data.get_image_height() - KERNEL_SIZE) / KERNEL_STRIDE + 1 # Number of filters that "fit" in a column of the input image
	num_fit_w = (data.get_image_width() - KERNEL_SIZE) / KERNEL_STRIDE + 1 # Number of filters that "fit" in a row of the input
	pooled_h = (num_fit_h - POOL_SIZE) / POOL_STRIDE + 1
	pooled_w = (num_fit_w - POOL_SIZE) / POOL_STRIDE + 1
	return pooled_h * pooled_w

def init_params(zeros=False):
	num_classes = data.get_num_classes()
	num_channels = data.get_num_channels()
	kernel = init_matrix(KERNEL_SIZE, KERNEL_SIZE, num_channels, zeros=zeros)
	conv_bias = 0.0
	fc_size = calc_fc_size(num_channels)
	fc_weights = init_2d_matrix(num_classes, fc_size)
	fc_bias = np.zeros((num_classes, 1))
	return kernel, conv_bias, fc_weights, fc_bias

def init_gradients():
	return init_params(zeros=True)

# Learns from given batch data using mini-batch gradient descent and returns updated parameters.
def update_params(batch, kernel, conv_bias, fc_weights, fc_bias, learning_rate):
	num_correct = 0
	num_classes = data.get_num_classes()
	num_examples = len(batch)
	target_matrix = np.zeros((num_examples, num_classes))
	pred_matrix = np.zeros((num_examples, num_classes))
	d_kernel, d_conv_bias, d_fc_weights, d_fc_bias = init_gradients()
	batch_loss = 0.0
	for i in xrange(len(batch)):
		img = batch[i]['image']
		label = batch[i]['label']
		target_matrix[i][label] = 1.0
		label_vector = np.zeros(num_classes)
		label_vector[label] = 1.0
		prediction, probs, layers = predict_single_example(img, label, num_classes, kernel, conv_bias, fc_weights, fc_bias)
		for class_num in xrange(num_classes): pred_matrix[i, class_num] = probs[class_num] # update prediction matrix
		if prediction == label: num_correct += 1
		[single_d_kernel, single_d_conv_bias, single_d_fc_weights, single_d_fc_bias] = learn_gradients(probs, label_vector, layers, fc_weights, fc_bias, img, kernel)
		d_kernel += single_d_kernel
		d_conv_bias += single_d_conv_bias
		d_fc_weights += single_d_fc_weights
		d_fc_bias += single_d_fc_bias
		batch_loss += cross_entropy(probs, label_vector)
	params = {'kernel' : kernel - d_kernel * learning_rate, 'conv_bias' : conv_bias - d_conv_bias * learning_rate, 'fc_weights' : fc_weights - d_fc_weights * learning_rate, 'fc_bias' : fc_bias - d_fc_bias * learning_rate}
	return params, num_correct, batch_loss


def predict_single_example(img, label, num_classes, kernel, conv_bias, fc_weights, fc_bias):
	conv_layer = convolve(img, kernel, conv_bias)
	conv_layer[conv_layer <= 0.0] = 0.0 #ReLU
	pooled_layer = maxpool(conv_layer, POOL_SIZE, POOL_STRIDE)
	fc_layer = np.reshape(pooled_layer, (-1, 1))
	fc_layer[fc_layer <= 0.0] = 0.0 #ReLU
	output_layer = fc_weights.dot(fc_layer) + fc_bias
	probs = softmax(output_layer.flatten())
	prediction = np.argmax(probs)
	layers = {'conv' : conv_layer, 'pooled' : pooled_layer, 'fc' : fc_layer}
	return prediction, probs, layers
