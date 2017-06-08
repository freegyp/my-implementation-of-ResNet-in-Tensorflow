# my-implementation-of-ResNet-in-Tensorflow
I have implemented a 34-Layer ResNet in Tensorflow Recently using Cifar-10 datasets and when the training accuracy reaches 99%, the testing accuracy stayed at about 78%. I'm new to Tensorflow and I'm hoping that someone can help me to check if there is any problems.

The running process is as follows:

1. Install Tensorflow.

2. Under the same folder as myResnet.py, download and extract https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz.

3. Run "pip install scikit-image"

4. Run "python myResnet.py"


I trained with all 50000 training images with each batch 100 images. Before each epoch (500 batches), I shuffled all 50000 training images.

When training the network, I applied random up/down,left/right flips and random saturation and brightness.

As the paper suggested, I used momentum optimizer with Nesterov's accelerated gradient, the momentum is 0.9 and the learning rate starts at 0.1 and gets divided by 10 when the epochs reaches 50 and 100. And I applied weight decay of 0.0001 to all the trainable weights, not including biases.

Here's the link to the paper:

https://arxiv.org/pdf/1512.03385.pdf

Did I miss anything when I implement the neural network? How should I improve?

I am a beginner on deep learning and I really appreciate any expert suggestions on improving this implementation.
