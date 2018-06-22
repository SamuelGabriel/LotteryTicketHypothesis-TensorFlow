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
parser.add_argument('--trainingsteps', type=int, default=18000)
parser.add_argument('--firstlayersize', type=int, default=300)
parser.add_argument('--secondlayersize', type=int, default=100)
parser.add_argument('--remainingcounts', type=str,
                    default='[(round(share*784*300), round(share*300*100), round(share*100*10)) ' +
                    'for share in ((.8**2)**i for i in range(1, 11))]')
args = parser.parse_args()
assert(args.resultdir is not None)
ITERATIVE = args.iterative
LOG_STEP = 100
NUM_ITERATIONS = args.trainingsteps
RESULT_PATH_PREFIX = args.resultdir
SAMPLE_START = args.samplestart
SAMPLE_SUPREMUM = args.samplesupremum
FIRST_LAYER_SIZE = args.firstlayersize
SECOND_LAYER_SIZE = args.secondlayersize
REMAINING_COUNTS = eval(args.remainingcounts)

print('Results written to {}'.format(RESULT_PATH_PREFIX))
print('Iterative Run {}'.format(ITERATIVE))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


# Network definition
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

prune = []
prune.append(tf.placeholder(tf.float32, shape=[784, FIRST_LAYER_SIZE]))
prune.append(tf.placeholder(tf.float32, shape=[
             FIRST_LAYER_SIZE, SECOND_LAYER_SIZE]))
prune.append(tf.placeholder(tf.float32, shape=[SECOND_LAYER_SIZE, 10]))

W = []
b = []
h_layer = []

W.append(weight_variable([784, FIRST_LAYER_SIZE]))
b.append(bias_variable([FIRST_LAYER_SIZE]))
h_layer.append(tf.nn.relu(tf.matmul(x, W[0] * prune[0]) + b[0]))

W.append(weight_variable([FIRST_LAYER_SIZE, SECOND_LAYER_SIZE]))
b.append(bias_variable([SECOND_LAYER_SIZE]))
h_layer.append(tf.nn.relu(tf.matmul(h_layer[0], W[1] * prune[1]) + b[1]))

W.append(weight_variable([SECOND_LAYER_SIZE, 10]))
b.append(bias_variable([10]))
y = tf.matmul(h_layer[1], W[2] * prune[2]) + b[2]

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train(p):
    iterations = []
    accuracies = []
    for i in range(NUM_ITERATIONS + 1):
        batch = mnist.train.next_batch(100)
        if i % LOG_STEP == 0:
            acc = accuracy.eval(feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels,
                **{p_ph: pr for p_ph, pr in zip(prune, p)}
            })
            accuracies.append(acc)
            iterations.append(i)
            print("test accuracy in step {}: {}" .format(i, acc))
        train_step.run(feed_dict={
            x: batch[0],
            y_: batch[1],
            **{p_ph: pr for p_ph, pr in zip(prune, p)}
        })
    return accuracies, iterations


def create_prune(x, remaining_count):
    '''
    Prune prune_share entries of all entries in this layer
    '''
    shape = x.shape
    flat_abs = np.abs(x.reshape(-1))
    argsort = flat_abs.argsort()
    prune_count = x.size-remaining_count
    pruner = np.ones(flat_abs.shape)
    pruner[argsort[:prune_count]] = 0
    pruner = pruner.reshape(shape)
    return pruner


def evaluate_sample(sample_n):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    W_init = [w.eval() for w in W]

    base_accuracies, base_iterations = train(
        [np.ones([784, FIRST_LAYER_SIZE]),
         np.ones([FIRST_LAYER_SIZE, SECOND_LAYER_SIZE]),
         np.ones([SECOND_LAYER_SIZE, 10])]
    )

    np.save('{}/base_accuracies_{}.npy'.format(
        RESULT_PATH_PREFIX, sample_n), base_accuracies)
    np.save('{}/base_iterations_{}.npy'.format(
        RESULT_PATH_PREFIX, sample_n), base_iterations)

    W_after_train = [w.eval() for w in W]

    p = [np.ones([784, FIRST_LAYER_SIZE]),
         np.ones([FIRST_LAYER_SIZE, SECOND_LAYER_SIZE]),
         np.ones([SECOND_LAYER_SIZE, 10])]

    for counts in REMAINING_COUNTS:
        if ITERATIVE:
            p = [create_prune(w.eval()*pr, c)
                 for w, pr, c in zip(W, p, counts)]
        else:
            p = [create_prune(w_after_train, c)
                 for w_after_train, c in zip(W_after_train, counts)]

        sess.run(tf.global_variables_initializer())
        for w, w_init in zip(W, W_init):
            sess.run(w.assign(w_init))

        accuracies, iterations = train(p)

        np.save(
            '{}/accuracies_sample_{}_restcounts_{}.npy'
            .format(RESULT_PATH_PREFIX, sample_n, counts), accuracies)
        np.save(
            '{}/iterations_sample_{}_restcounts_{}.npy'
            .format(RESULT_PATH_PREFIX, sample_n, counts), iterations)


for sample_it in range(SAMPLE_START, SAMPLE_SUPREMUM):
    evaluate_sample(sample_it)
