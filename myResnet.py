import os
#import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

import numpy as np
from skimage import io,transform

import cPickle

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

complete_data = unpickle("/notebooks/cifar-10-batches-py/data_batch_1")
complete_imgs, complete_tags = complete_data['data'], complete_data['labels']
test_data = unpickle("/notebooks/cifar-10-batches-py/test_batch")
test_imgs, test_tags = test_data['data'], test_data['labels']

def getBatch(idx):
	imgs = np.ndarray([100,224,224,3])
	tags = np.ndarray([100,10])
	for i in xrange(100):
		img0 = complete_imgs[idx*100+i].reshape([3,32,32,1])
		img = np.concatenate((img0[0],img0[1],img0[2]),2)
		img = transform.resize(img,[224,224,3])
		tag = np.zeros([10])
		tag[complete_tags[idx*100+i]] = 1
		imgs[i], tags[i] = img, tag
	return imgs,tags

def getTestBatch(idx):
	imgs = np.ndarray([100,224,224,3])
	tags = np.ndarray([100,10])
	for i in xrange(100):
		img0 = test_imgs[idx*100+i].reshape([3,32,32,1])
		img = np.concatenate((img0[0],img0[1],img0[2]),2)
		img = transform.resize(img,[224,224,3])
		tag = np.zeros([10])
		tag[test_tags[idx*100+i]] = 1
		imgs[i], tags[i] = img, tag
	return imgs,tags

x = tf.placeholder(tf.float32, [None,224,224,3])
y = tf.placeholder(tf.float32, [None,10])

"""

	Convolution Layers 0

"""
conv_weights_0 = tf.Variable(tf.random_normal([7,7,3,64]),dtype=tf.float32)
conv_0 = tf.nn.conv2d(x, conv_weights_0, strides=[1,2,2,1], padding='SAME')

axis = list(range(len(conv_0.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_0, axis)

beta = tf.Variable(tf.zeros(conv_0.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_0.get_shape()[-1:]),dtype=tf.float32)

conv_0 = tf.nn.batch_normalization(conv_0, mean, variance, beta, gamma, 0.001)

conv_0 = tf.nn.relu(conv_0)

conv_0_out = tf.nn.max_pool(conv_0, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')



"""
	Convolution Layers 1 starts here.
	Convolution Layers 1, Sub Unit 0

"""

conv_weights_1_0 = tf.Variable(tf.random_normal([3,3,64,64]),dtype=tf.float32)
conv_1_0 = tf.nn.conv2d(conv_0_out, conv_weights_1_0, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_1_0.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_1_0, axis)

beta = tf.Variable(tf.zeros(conv_1_0.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_1_0.get_shape()[-1:]),dtype=tf.float32)

conv_1_0 = tf.nn.batch_normalization(conv_1_0, mean, variance, beta, gamma, 0.001)

conv_1_0 = tf.nn.relu(conv_1_0)

conv_weights_1_1 = tf.Variable(tf.random_normal([3,3,64,64]),dtype=tf.float32)
conv_1_1 = tf.nn.conv2d(conv_1_0, conv_weights_1_1, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_1_1.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_1_1, axis)

beta = tf.Variable(tf.zeros(conv_1_1.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_1_1.get_shape()[-1:]),dtype=tf.float32)

conv_1_1 = tf.nn.batch_normalization(conv_1_1, mean, variance, beta, gamma, 0.001)

conv_1_1 = conv_1_1 + conv_0_out

conv_1_1 = tf.nn.relu(conv_1_1)

"""

	Convolution Layers 1, Sub Unit 1

"""

conv_weights_1_2 = tf.Variable(tf.random_normal([3,3,64,64]),dtype=tf.float32)
conv_1_2 = tf.nn.conv2d(conv_1_1, conv_weights_1_2, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_1_2.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_1_2, axis)

beta = tf.Variable(tf.zeros(conv_1_2.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_1_2.get_shape()[-1:]),dtype=tf.float32)

conv_1_2 = tf.nn.batch_normalization(conv_1_2, mean, variance, beta, gamma, 0.001)

conv_1_2 = tf.nn.relu(conv_1_2)

conv_weights_1_3 = tf.Variable(tf.random_normal([3,3,64,64]),dtype=tf.float32)
conv_1_3 = tf.nn.conv2d(conv_1_2, conv_weights_1_3, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_1_3.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_1_3, axis)

beta = tf.Variable(tf.zeros(conv_1_3.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_1_3.get_shape()[-1:]),dtype=tf.float32)

conv_1_3 = tf.nn.batch_normalization(conv_1_3, mean, variance, beta, gamma, 0.001)

conv_1_3 = conv_1_3 + conv_1_1

conv_1_3 = tf.nn.relu(conv_1_3)

"""

	Convolution Layers 1, Sub Unit 2

"""

conv_weights_1_4 = tf.Variable(tf.random_normal([3,3,64,64]),dtype=tf.float32)
conv_1_4 = tf.nn.conv2d(conv_1_3, conv_weights_1_4, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_1_4.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_1_4, axis)

beta = tf.Variable(tf.zeros(conv_1_4.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_1_4.get_shape()[-1:]),dtype=tf.float32)

conv_1_4 = tf.nn.batch_normalization(conv_1_4, mean, variance, beta, gamma, 0.001)

conv_1_4 = tf.nn.relu(conv_1_4)

conv_weights_1_5 = tf.Variable(tf.random_normal([3,3,64,64]),dtype=tf.float32)
conv_1_5 = tf.nn.conv2d(conv_1_4, conv_weights_1_5, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_1_5.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_1_5, axis)

beta = tf.Variable(tf.zeros(conv_1_5.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_1_5.get_shape()[-1:]),dtype=tf.float32)

conv_1_5 = tf.nn.batch_normalization(conv_1_5, mean, variance, beta, gamma, 0.001)

conv_1_5 = conv_1_5 + conv_1_3

conv_1_out = tf.nn.relu(conv_1_5)

# Convolution Layers 1 ends here.


"""
	Convolution Layers 2 starts here.
	Convolution Layers 2, Sub Unit 0

"""

conv_weights_2_0 = tf.Variable(tf.random_normal([3,3,64,128]),dtype=tf.float32)
conv_2_0 = tf.nn.conv2d(conv_1_out, conv_weights_2_0, strides=[1,2,2,1], padding="SAME")

axis = list(range(len(conv_2_0.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_0, axis)

beta = tf.Variable(tf.zeros(conv_2_0.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_2_0.get_shape()[-1:]),dtype=tf.float32)

conv_2_0 = tf.nn.batch_normalization(conv_2_0, mean, variance, beta, gamma, 0.001)

conv_2_0 = tf.nn.relu(conv_2_0)

conv_weights_2_1 = tf.Variable(tf.random_normal([3,3,128,128]),dtype=tf.float32)
conv_2_1 = tf.nn.conv2d(conv_2_0, conv_weights_2_1, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_2_1.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_1, axis)

beta = tf.Variable(tf.zeros(conv_2_1.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_2_1.get_shape()[-1:]),dtype=tf.float32)

conv_2_1 = tf.nn.batch_normalization(conv_2_1, mean, variance, beta, gamma, 0.001)

conv_weights_2_pre = tf.Variable(tf.random_normal([1,1,64,128]),dtype=tf.float32,trainable=False)
conv_2_pre = tf.nn.conv2d(conv_1_out, conv_weights_2_pre, strides=[1,2,2,1], padding="SAME")

axis = list(range(len(conv_2_pre.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_pre, axis)

conv_2_pre = tf.nn.batch_normalization(conv_2_pre, mean, variance, None, None, 0.001)

conv_2_1 = conv_2_1 + conv_2_pre

conv_2_1 = tf.nn.relu(conv_2_1)

"""

	Convolution Layers 2, Sub Unit 1

"""

conv_weights_2_2 = tf.Variable(tf.random_normal([3,3,128,128]),dtype=tf.float32)
conv_2_2 = tf.nn.conv2d(conv_2_1, conv_weights_2_2, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_2_2.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_2, axis)

beta = tf.Variable(tf.zeros(conv_2_2.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_2_2.get_shape()[-1:]),dtype=tf.float32)

conv_2_2 = tf.nn.batch_normalization(conv_2_2, mean, variance, beta, gamma, 0.001)

conv_2_2 = tf.nn.relu(conv_2_2)

conv_weights_2_3 = tf.Variable(tf.random_normal([3,3,128,128]),dtype=tf.float32)
conv_2_3 = tf.nn.conv2d(conv_2_2, conv_weights_2_3, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_2_3.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_3, axis)

beta = tf.Variable(tf.zeros(conv_2_3.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_2_3.get_shape()[-1:]),dtype=tf.float32)

conv_2_3 = tf.nn.batch_normalization(conv_2_3, mean, variance, beta, gamma, 0.001)

conv_2_3 = conv_2_3 + conv_2_1

conv_2_3 = tf.nn.relu(conv_2_3)

"""

	Convolution Layers 2, Sub Unit 2

"""

conv_weights_2_4 = tf.Variable(tf.random_normal([3,3,128,128]),dtype=tf.float32)
conv_2_4 = tf.nn.conv2d(conv_2_3, conv_weights_2_4, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_2_4.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_4, axis)

beta = tf.Variable(tf.zeros(conv_2_4.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_2_4.get_shape()[-1:]),dtype=tf.float32)

conv_2_4 = tf.nn.batch_normalization(conv_2_4, mean, variance, beta, gamma, 0.001)

conv_2_4 = tf.nn.relu(conv_2_4)

conv_weights_2_5 = tf.Variable(tf.random_normal([3,3,128,128]),dtype=tf.float32)
conv_2_5 = tf.nn.conv2d(conv_2_4, conv_weights_2_5, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_2_5.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_5, axis)

beta = tf.Variable(tf.zeros(conv_2_5.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_2_5.get_shape()[-1:]),dtype=tf.float32)

conv_2_5 = tf.nn.batch_normalization(conv_2_5, mean, variance, beta, gamma, 0.001)

conv_2_5 = conv_2_5 + conv_2_3

conv_2_5 = tf.nn.relu(conv_2_5)

"""

	Convolution Layers 2, Sub Unit 3

"""

conv_weights_2_6 = tf.Variable(tf.random_normal([3,3,128,128]),dtype=tf.float32)
conv_2_6 = tf.nn.conv2d(conv_2_5, conv_weights_2_6, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_2_6.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_6, axis)

beta = tf.Variable(tf.zeros(conv_2_6.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_2_6.get_shape()[-1:]),dtype=tf.float32)

conv_2_6 = tf.nn.batch_normalization(conv_2_6, mean, variance, beta, gamma, 0.001)

conv_2_6 = tf.nn.relu(conv_2_6)

conv_weights_2_7 = tf.Variable(tf.random_normal([3,3,128,128]),dtype=tf.float32)
conv_2_7 = tf.nn.conv2d(conv_2_6, conv_weights_2_7, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_2_7.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_2_7, axis)

beta = tf.Variable(tf.zeros(conv_2_7.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_2_7.get_shape()[-1:]),dtype=tf.float32)

conv_2_7 = tf.nn.batch_normalization(conv_2_7, mean, variance, beta, gamma, 0.001)

conv_2_7 = conv_2_7 + conv_2_5

conv_2_out = tf.nn.relu(conv_2_7)

# Convolution Layers 2 ends here.


"""
	Convolution Layers 3 starts here.
	Convolution Layers 3, Sub Unit 0

"""

conv_weights_3_0 = tf.Variable(tf.random_normal([3,3,128,256]),dtype=tf.float32)
conv_3_0 = tf.nn.conv2d(conv_2_out, conv_weights_3_0, strides=[1,2,2,1], padding="SAME")

axis = list(range(len(conv_3_0.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_0, axis)

beta = tf.Variable(tf.zeros(conv_3_0.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_0.get_shape()[-1:]),dtype=tf.float32)

conv_3_0 = tf.nn.batch_normalization(conv_3_0, mean, variance, beta, gamma, 0.001)

conv_3_0 = tf.nn.relu(conv_3_0)

conv_weights_3_1 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_1 = tf.nn.conv2d(conv_3_0, conv_weights_3_1, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_1.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_1, axis)

beta = tf.Variable(tf.zeros(conv_3_1.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_1.get_shape()[-1:]),dtype=tf.float32)

conv_3_1 = tf.nn.batch_normalization(conv_3_1, mean, variance, beta, gamma, 0.001)

conv_weights_3_pre = tf.Variable(tf.random_normal([1,1,128,256]),dtype=tf.float32,trainable=False)
conv_3_pre = tf.nn.conv2d(conv_2_out, conv_weights_3_pre, strides=[1,2,2,1], padding="SAME")

axis = list(range(len(conv_3_pre.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_pre, axis)

conv_3_pre = tf.nn.batch_normalization(conv_3_pre, mean, variance, None, None, 0.001)

conv_3_1 = conv_3_1 + conv_3_pre

conv_3_1 = tf.nn.relu(conv_3_1)

"""

	Convolution Layers 3, Sub Unit 1

"""

conv_weights_3_2 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_2 = tf.nn.conv2d(conv_3_1, conv_weights_3_2, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_2.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_2, axis)

beta = tf.Variable(tf.zeros(conv_3_2.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_2.get_shape()[-1:]),dtype=tf.float32)

conv_3_2 = tf.nn.batch_normalization(conv_3_2, mean, variance, beta, gamma, 0.001)

conv_3_2 = tf.nn.relu(conv_3_2)

conv_weights_3_3 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_3 = tf.nn.conv2d(conv_3_2, conv_weights_3_3, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_3.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_3, axis)

beta = tf.Variable(tf.zeros(conv_3_3.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_3.get_shape()[-1:]),dtype=tf.float32)

conv_3_3 = tf.nn.batch_normalization(conv_3_3, mean, variance, beta, gamma, 0.001)

conv_3_3 = conv_3_3 + conv_3_1

conv_3_3 = tf.nn.relu(conv_3_3)

"""

	Convolution Layers 3, Sub Unit 2

"""

conv_weights_3_4 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_4 = tf.nn.conv2d(conv_3_3, conv_weights_3_4, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_4.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_4, axis)

beta = tf.Variable(tf.zeros(conv_3_4.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_4.get_shape()[-1:]),dtype=tf.float32)

conv_3_4 = tf.nn.batch_normalization(conv_3_4, mean, variance, beta, gamma, 0.001)

conv_3_4 = tf.nn.relu(conv_3_4)

conv_weights_3_5 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_5 = tf.nn.conv2d(conv_3_4, conv_weights_3_5, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_5.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_5, axis)

beta = tf.Variable(tf.zeros(conv_3_5.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_5.get_shape()[-1:]),dtype=tf.float32)

conv_3_5 = tf.nn.batch_normalization(conv_3_5, mean, variance, beta, gamma, 0.001)

conv_3_5 = conv_3_5 + conv_3_3

conv_3_5 = tf.nn.relu(conv_3_5)

"""

	Convolution Layers 3, Sub Unit 3

"""

conv_weights_3_6 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_6 = tf.nn.conv2d(conv_3_5, conv_weights_3_6, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_6.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_6, axis)

beta = tf.Variable(tf.zeros(conv_3_6.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_6.get_shape()[-1:]),dtype=tf.float32)

conv_3_6 = tf.nn.batch_normalization(conv_3_6, mean, variance, beta, gamma, 0.001)

conv_3_6 = tf.nn.relu(conv_3_6)

conv_weights_3_7 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_7 = tf.nn.conv2d(conv_3_6, conv_weights_3_7, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_7.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_7, axis)

beta = tf.Variable(tf.zeros(conv_3_7.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_7.get_shape()[-1:]),dtype=tf.float32)

conv_3_7 = tf.nn.batch_normalization(conv_3_7, mean, variance, beta, gamma, 0.001)

conv_3_7 = conv_3_7 + conv_3_5

conv_3_7 = tf.nn.relu(conv_3_7)

"""

	Convolution Layers 3, Sub Unit 4

"""

conv_weights_3_8 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_8 = tf.nn.conv2d(conv_3_7, conv_weights_3_8, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_8.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_8, axis)

beta = tf.Variable(tf.zeros(conv_3_8.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_8.get_shape()[-1:]),dtype=tf.float32)

conv_3_8 = tf.nn.batch_normalization(conv_3_8, mean, variance, beta, gamma, 0.001)

conv_3_8 = tf.nn.relu(conv_3_8)

conv_weights_3_9 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_9 = tf.nn.conv2d(conv_3_8, conv_weights_3_9, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_9.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_9, axis)

beta = tf.Variable(tf.zeros(conv_3_9.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_9.get_shape()[-1:]),dtype=tf.float32)

conv_3_9 = tf.nn.batch_normalization(conv_3_9, mean, variance, beta, gamma, 0.001)

conv_3_9 = conv_3_9 + conv_3_7

conv_3_9 = tf.nn.relu(conv_3_9)

"""

	Convolution Layers 3, Sub Unit 5

"""

conv_weights_3_10 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_10 = tf.nn.conv2d(conv_3_9, conv_weights_3_10, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_10.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_10, axis)

beta = tf.Variable(tf.zeros(conv_3_10.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_10.get_shape()[-1:]),dtype=tf.float32)

conv_3_10 = tf.nn.batch_normalization(conv_3_10, mean, variance, beta, gamma, 0.001)

conv_3_10 = tf.nn.relu(conv_3_10)

conv_weights_3_11 = tf.Variable(tf.random_normal([3,3,256,256]),dtype=tf.float32)
conv_3_11 = tf.nn.conv2d(conv_3_10, conv_weights_3_11, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_3_11.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_3_11, axis)

beta = tf.Variable(tf.zeros(conv_3_11.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_3_11.get_shape()[-1:]),dtype=tf.float32)

conv_3_11 = tf.nn.batch_normalization(conv_3_11, mean, variance, beta, gamma, 0.001)

conv_3_11 = conv_3_11 + conv_3_9

conv_3_out = tf.nn.relu(conv_3_11)

# Convolution Layers 3 ends here.


"""
	Convolution Layers 4 starts here.
	Convolution Layers 4, Sub Unit 0

"""

conv_weights_4_0 = tf.Variable(tf.random_normal([3,3,256,512]),dtype=tf.float32)
conv_4_0 = tf.nn.conv2d(conv_3_out, conv_weights_4_0, strides=[1,2,2,1], padding="SAME")

axis = list(range(len(conv_4_0.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_4_0, axis)

beta = tf.Variable(tf.zeros(conv_4_0.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_4_0.get_shape()[-1:]),dtype=tf.float32)

conv_4_0 = tf.nn.batch_normalization(conv_4_0, mean, variance, beta, gamma, 0.001)

conv_4_0 = tf.nn.relu(conv_4_0)

conv_weights_4_1 = tf.Variable(tf.random_normal([3,3,512,512]),dtype=tf.float32)
conv_4_1 = tf.nn.conv2d(conv_4_0, conv_weights_4_1, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_4_1.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_4_1, axis)

beta = tf.Variable(tf.zeros(conv_4_1.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_4_1.get_shape()[-1:]),dtype=tf.float32)

conv_4_1 = tf.nn.batch_normalization(conv_4_1, mean, variance, beta, gamma, 0.001)

conv_weights_4_pre = tf.Variable(tf.random_normal([1,1,256,512]),dtype=tf.float32,trainable=False)
conv_4_pre = tf.nn.conv2d(conv_3_out, conv_weights_4_pre, strides=[1,2,2,1], padding="SAME")

axis = list(range(len(conv_4_pre.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_4_pre, axis)

conv_4_pre = tf.nn.batch_normalization(conv_4_pre, mean, variance, None, None, 0.001)

conv_4_1 = conv_4_1 + conv_4_pre

conv_4_1 = tf.nn.relu(conv_4_1)

"""

	Convolution Layers 4, Sub Unit 1

"""

conv_weights_4_2 = tf.Variable(tf.random_normal([3,3,512,512]),dtype=tf.float32)
conv_4_2 = tf.nn.conv2d(conv_4_1, conv_weights_4_2, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_4_2.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_4_2, axis)

beta = tf.Variable(tf.zeros(conv_4_2.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_4_2.get_shape()[-1:]),dtype=tf.float32)

conv_4_2 = tf.nn.batch_normalization(conv_4_2, mean, variance, beta, gamma, 0.001)

conv_4_2 = tf.nn.relu(conv_4_2)

conv_weights_4_3 = tf.Variable(tf.random_normal([3,3,512,512]),dtype=tf.float32)
conv_4_3 = tf.nn.conv2d(conv_4_2, conv_weights_4_3, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_4_3.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_4_3, axis)

beta = tf.Variable(tf.zeros(conv_4_3.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_4_3.get_shape()[-1:]),dtype=tf.float32)

conv_4_3 = tf.nn.batch_normalization(conv_4_3, mean, variance, beta, gamma, 0.001)

conv_4_3 = conv_4_3 + conv_4_1

conv_4_3 = tf.nn.relu(conv_4_3)

"""

	Convolution Layers 4, Sub Unit 2

"""

conv_weights_4_4 = tf.Variable(tf.random_normal([3,3,512,512]),dtype=tf.float32)
conv_4_4 = tf.nn.conv2d(conv_4_3, conv_weights_4_4, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_4_4.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_4_4, axis)

beta = tf.Variable(tf.zeros(conv_4_4.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_4_4.get_shape()[-1:]),dtype=tf.float32)

conv_4_4 = tf.nn.batch_normalization(conv_4_4, mean, variance, beta, gamma, 0.001)

conv_4_4 = tf.nn.relu(conv_4_4)

conv_weights_4_5 = tf.Variable(tf.random_normal([3,3,512,512]),dtype=tf.float32)
conv_4_5 = tf.nn.conv2d(conv_4_4, conv_weights_4_5, strides=[1,1,1,1], padding="SAME")

axis = list(range(len(conv_4_5.get_shape()) - 1))
mean, variance = tf.nn.moments(conv_4_5, axis)

beta = tf.Variable(tf.zeros(conv_4_5.get_shape()[-1:]),dtype=tf.float32)
gamma = tf.Variable(tf.ones(conv_4_5.get_shape()[-1:]),dtype=tf.float32)

conv_4_5 = tf.nn.batch_normalization(conv_4_5, mean, variance, beta, gamma, 0.001)

conv_4_5 = conv_4_5 + conv_4_3

conv_4_out = tf.nn.relu(conv_4_5)

# Convolution Layers 4 ends here.

"""

	Post convolution networks starts here.

"""

fc_pre = tf.nn.avg_pool(conv_4_out, ksize=[1,7,7,1], strides=[1,1,1,1], padding="SAME")
fc_pre = tf.reshape(fc_pre,[-1,7*7*512])
fc_weight = tf.Variable(tf.random_normal([7*7*512,1000]),dtype=tf.float32)
fc_bias = tf.Variable(tf.random_normal([1000]),dtype=tf.float32)

fc1 = tf.add(tf.matmul(fc_pre, fc_weight),fc_bias)

fc2_weight = tf.Variable(tf.random_normal([1000,10]),dtype=tf.float32)
fc2_bias = tf.Variable(tf.random_normal([10]),dtype=tf.float32)

out = tf.add(tf.matmul(fc1,fc2_weight),fc2_bias)

pred = tf.nn.softmax(out)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

"""

	Now we can run our graph.

"""

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in xrange(100):
		print "getting batch {0}".format(i)
		batch_x, batch_y = getBatch(i)
		print "Started to train batch {0}".format(i)
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		print i
		testBatchX, testBatchY = getTestBatch(i)
		loss, acc, cor_pred = sess.run([cost, accuracy, pred], feed_dict={x: testBatchX, y: testBatchY})
		print("Minibatch Loss= " + \
   	          "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
		print cor_pred










