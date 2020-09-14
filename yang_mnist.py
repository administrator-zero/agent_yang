# -*- coding: UTF-8 -*-
import numpy as np
from datetime import datetime
from image_loader import ImageLoader
from label_loader import LabelLoader
# from yang_net import YangNet
from bp_net import YangNet
# from numba import vectorize
# @vectorize(['float32(float32, float32)'], target='cuda')

def get_training_data_set():
    """
    Get the train data set.
    """
    image_loader = ImageLoader('data/train-images.idx3-ubyte', 60000)
    label_loader = LabelLoader('data/train-labels.idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    """
    Get the test data set.
    """
    image_loader = ImageLoader('data/t10k-images.idx3-ubyte', 10000)
    label_loader = LabelLoader('data/t10k-labels.idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()


def get_result(vec):
    max_value_index = 0
    max_value = 0

    # print("vec", vec)
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        temp = network.predict(test_data_set[i])
        predict = get_result(temp)
        if label != predict:
            error += 1
    return float(error) / float(total)


def now():
    return datetime.now().strftime('%c')


def normalization(input_x):
    f_put = np.array(input_x).flatten()
    max_in = max(f_put)
    min_in = min(f_put)
    return (input_x - min_in) / (max_in - min_in)


def transpose(args):
    args1 = [np.array(line).reshape(len(line), 1) for line in args[1]]
    args0 = [np.expand_dims(line, axis=0) for line in args[0]]
    return args0, args1


def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set_data, train_labels = transpose(get_training_data_set())
    test_data_set_data, test_labels = transpose(get_test_data_set())
    # import cv2
    # a = np.array(train_data_set_data)
    # cv2.imshow('1', a[0, 0, :, :])
    # cv2.waitKey(0)

    train_data_set_data = normalization(np.array(train_data_set_data))
    test_data_set_data = normalization(np.array(test_data_set_data))
    network = YangNet()
    # network = BPNet()
    while True:
        epoch += 1
        print(now(), 'epoch:', epoch)
        network.train(train_labels, train_data_set_data, 0.01, 1)
        print('%s epoch %d finished, loss %f' %
              (now(), epoch, network.loss(train_labels[-1], network.predict(train_data_set_data[-1]))))
        if epoch % 1 == 0:
            error_ratio = evaluate(network, test_data_set_data, test_labels)
            print('%s after epoch %d, error ratio is %f, correct radio is %.2f%%' %
                  (now(), epoch, error_ratio, (1 - error_ratio) * 100))
            if error_ratio > last_error_ratio:
                # break
                print("done")
            else:
                last_error_ratio = error_ratio


if __name__ == '__main__':
    train_and_evaluate()
