from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from multiprocessing import Pool
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import argparse

parser = argparse.ArgumentParser("Arguments")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument("--iterative", default=True, type=str2bool)
parser.add_argument("--resultdir", type=str)
parser.add_argument("--samplestart", type=int, default=0)
parser.add_argument("--samplesupremum", type=int, default=5)
parser.add_argument('--reductionsteps', type=int, default=10)
parser.add_argument('--trainingsteps', type=int, default=18000)
args = parser.parse_args()
assert(args.resultdir is not None)
ITERATIVE = args.iterative
LOG_STEP = 100
NUM_ITERATIONS = args.trainingsteps
REST_SHARE_REDUCER = 0.8 ** 2
RESULT_PATH_PREFIX = args.resultdir
REDUCTION_STEPS = args.reductionsteps
SAMPLE_START = args.samplestart
SAMPLE_SUPREMUM = args.samplesupremum

print('Results written to {}'.format(RESULT_PATH_PREFIX))
print('Iterative Run {}'.format(ITERATIVE))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
prune_1 = tf.placeholder(tf.float32, shape=[784, 300])
prune_2 = tf.placeholder(tf.float32, shape=[300, 100])
prune_3 = tf.placeholder(tf.float32, shape=[100, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


W_1 = weight_variable([784, 300])
b_1 = bias_variable([300])
h_layer1 = tf.nn.relu(tf.matmul(x, W_1 * prune_1) + b_1)

W_2 = weight_variable([300, 100])
b_2 = bias_variable([100])
h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W_2 * prune_2) + b_2)

W_3 = weight_variable([100, 10])
b_3 = bias_variable([10])
y = tf.matmul(h_layer2, W_3 * prune_3) + b_3

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train(p_1, p_2, p_3):
    iterations = []
    accuracies = []
    for i in range(NUM_ITERATIONS + 1):
        batch = mnist.train.next_batch(100)
        if i % LOG_STEP == 0:
            acc = accuracy.eval(feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels,
                prune_1: p_1,
                prune_2: p_2,
                prune_3: p_3})
            accuracies.append(acc)
            iterations.append(i)
            print("test accuracy in step {}: {}" .format(i, acc))
        train_step.run(feed_dict={
            x: batch[0],
            y_: batch[1],
            prune_1: p_1,
            prune_2: p_2,
            prune_3: p_3
        })
    return accuracies, iterations


def create_prune(x, prune_share):
    '''
    Prune prune_share entries of all entries in this layer
    '''
    shape = x.shape
    flat_abs = np.abs(x.reshape(-1))
    argsort = flat_abs.argsort()
    prune_count = round(len(flat_abs) * prune_share)
    pruner = np.ones(flat_abs.shape)
    pruner[argsort[:prune_count]] = 0
    pruner = pruner.reshape(shape)
    return pruner


def evaluate_sample(sample_n):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    W_1_init = W_1.eval()
    W_2_init = W_2.eval()
    W_3_init = W_3.eval()

    base_accuracies, base_iterations = train(
        np.ones([784, 300]), np.ones([300, 100]), np.ones([100, 10]))

    np.save('{}/base_accuracies_{}.npy'.format(
        RESULT_PATH_PREFIX, sample_n), base_accuracies)
    np.save('{}/base_iterations_{}.npy'.format(
        RESULT_PATH_PREFIX, sample_n), base_iterations)

    rest_share = 1.

    W_1_after_train = W_1.eval()
    W_2_after_train = W_2.eval()
    W_3_after_train = W_3.eval()

    p_1 = np.ones([784, 300])
    p_2 = np.ones([300, 100])
    p_3 = np.ones([100, 10])

    for step in range(REDUCTION_STEPS):
        rest_share *= REST_SHARE_REDUCER
        if ITERATIVE:
            p_1 = create_prune(W_1.eval()*p_1, 1. - rest_share)
            p_2 = create_prune(W_2.eval()*p_2, 1. - rest_share)
            p_3 = create_prune(W_3.eval()*p_3, (1. - rest_share)/2)
        else:
            p_1 = create_prune(W_1_after_train, 1. - rest_share)
            p_2 = create_prune(W_2_after_train, 1. - rest_share)
            p_3 = create_prune(W_3_after_train, (1. - rest_share)/2)

        sess.run(tf.global_variables_initializer())
        sess.run(W_1.assign(W_1_init))
        sess.run(W_2.assign(W_2_init))
        sess.run(W_3.assign(W_3_init))

        print("Remaining share of weights: {}".format(rest_share))
        accuracies, iterations = train(p_1, p_2, p_3)

        np.save(
            '{}/accuracies_sample_{}_restshare_step_{}.npy'.format(RESULT_PATH_PREFIX, sample_n, step), accuracies)
        np.save(
            '{}/iterations_sample_{}_restshare_step_{}.npy'.format(RESULT_PATH_PREFIX, sample_n, step), iterations)

    for control_ex in range(3):
        sess.run(tf.global_variables_initializer())
        sess.run(W_1.assign(W_1_init))
        sess.run(W_2.assign(W_2_init))
        sess.run(W_3.assign(W_3_init))

        p_1 = np.ones([784, 300])
        p_2 = np.ones([300, 100])
        p_3 = np.ones([100, 10])
        rest_share = 1.

        for step in range(REDUCTION_STEPS):
            rest_share *= REST_SHARE_REDUCER

            if ITERATIVE:
                p_1 = create_prune(W_1.eval()*p_1, 1. - rest_share)
                p_2 = create_prune(W_2.eval()*p_2, 1. - rest_share)
                p_3 = create_prune(W_3.eval()*p_3, (1. - rest_share)/2)
            else:
                p_1 = create_prune(W_1_after_train, 1. - rest_share)
                p_2 = create_prune(W_2_after_train, 1. - rest_share)
                p_3 = create_prune(W_3_after_train, (1. - rest_share)/2)

            sess.run(tf.global_variables_initializer())

            print("Remaining share of weights: {}".format(rest_share))
            accuracies, iterations = train(p_1, p_2, p_3)

            np.save(
                '{}/accuracies_sample_{}_control_{}_restshare_step_{}.npy'.format(RESULT_PATH_PREFIX, sample_n, control_ex, step), accuracies)
            np.save(
                '{}/iterations_sample_{}_control_{}_restshare_step_{}.npy'.format(RESULT_PATH_PREFIX, sample_n, control_ex, step), iterations)


# pool = Pool(SAMPLE_SUPREMUM-SAMPLE_START-1)
# pool.map(evaluate_sample, range(SAMPLE_START, SAMPLE_SUPREMUM))
for sample_it in range(SAMPLE_START, SAMPLE_SUPREMUM):
    evaluate_sample(sample_it)
