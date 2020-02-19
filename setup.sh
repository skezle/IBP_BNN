mkdir -p src/logs
mkdir -p src/plots
mkdir -p src/results
mkdir -p src/data
mkdir -p src/data/mnist_background_images
mkdir -p src/data/mnist_background_random
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip -P src/data/mnist_background_images
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip -P src/data/mnist_background_random
unzip src/data/mnist_background_images/mnist_background_images.zip -d src/data/mnist_background_images
unzip src/data/mnist_background_random/mnist_background_random.zip -d src/data/mnist_background_random

wget http://deeplearning.net/data/mnist/mnist.pkl.gz -P src/data