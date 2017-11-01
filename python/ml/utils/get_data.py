import mxnet as mx
import zipfile
import os
import numpy as np
import struct

__UTILS_PATH__ = os.path.dirname(os.path.abspath(__file__))
__DATASET_PATH__ = os.path.join(__UTILS_PATH__, '..', '..', '..', 'dataset')

def get_mnist_iter(batch_size, data_name = 'data', label_name = 'softmax_label'):
    """Read and return NDArrayIter of mnist dataset

    Parameters:
    __________

    batch_size : int
    data_name : str
              data name of iterator. If none, default name 'data' of NDArrayIter will be used.
    label_name : str
              label name of iterator. If noe, default name 'softmax_label' of NDArrayIter will be used.

    Returns
    _______

    Tuple of NDArrayIter
              First element will be train iter, and second will be test iter.
    """

    zippath = os.path.join(__DATASET_PATH__, 'mnist.zip')
    datapath = os.path.join(__DATASET_PATH__, 'mnist')
    if not os.path.exists(datapath):
        os.makedirs(datapath)

    cwd = os.getcwd()
    os.chdir(datapath)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
           zf = zipfile.ZipFile(zippath, 'r')
           zf.extractall()
           zf.close()

    def read_data(label_path, image_path):
        with open(label_path, 'rb') as file:
            struct.unpack('>II', file.read(8))
            label = np.fromstring(file.read(), dtype = np.int8)

        with open(image_path, 'rb') as file:
            _, _, rows, cols = struct.unpack('>IIII', file.read(16))
            image = np.fromstring(file.read(), dtype = np.int8).reshape(
                    len(label), rows, cols)
            image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32) / 255

        return (label, image)

    (train_label, train_image) = read_data('train-labels-idx1-ubyte',
            'train-images-idx3-ubyte')

    (test_label, test_image) = read_data('t10k-labels-idx1-ubyte',
            't10k-images-idx3-ubyte')

    os.chdir(cwd)

    train_iter = mx.io.NDArrayIter(train_image, train_label, batch_size = batch_size,
            data_name = data_name, label_name = label_name, shuffle = True)
    test_iter = mx.io.NDArrayIter(test_image, test_label, batch_size = batch_size, 
            data_name = data_name, label_name = label_name, shuffle = False)

    return (train_iter, test_iter)


if __name__ == '__main__':
    mnist_train, mnist_test = get_mnist_iter(32)
    print(get_mnist_iter.__doc__)

    print('Number of train data: %d' % mnist_train.num_data)
    print('Number of test data: %d' % mnist_test.num_data)

