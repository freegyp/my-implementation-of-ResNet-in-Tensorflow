# my-implementation-of-ResNet-in-Tensorflow
I have implemented a 34-Layer ResNet in Tensorflow Recently using Cifar-10 datasets but the training accuracy was not as high as I expected. I'm new to Tensorflow and I'm hoping that someone can help me to check if there is any problems.

The running process is as follows:

1. Install Tensorflow.

2. Under the same folder as myResnet.py, download and extract https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz.

3. Run "pip install scikit-image"

4. Run "python myResnet.py"

The validation accuracy I got was about 65%, as the following graph.


![image](/test_acc.png)


I trained with all 50000 training images with each batch 100 images. Before each epoch (500 batches), I shuffled all 50000 training images.

I used AdamOptimizer with learning rate of 0.001 on training the network.

Did I miss anything when I implement the neural network? How should I improve?
