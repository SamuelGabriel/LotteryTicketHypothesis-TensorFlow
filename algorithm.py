from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
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
parser.add_argument('--layerrepetitions', type=str, default='[1,1]')
parser.add_argument('--remainingcounts', type=str,
                    default='[((round(share*784*300),), (round(share*300*100),), (round(share*100*10),)) ' +
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
LAYER_REPETITIONS = eval(args.layerrepetitions)

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
prune.append(
    [tf.placeholder(tf.float32, shape=[784, FIRST_LAYER_SIZE])] *
    LAYER_REPETITIONS[0]
)
prune.append(
    [tf.placeholder(tf.float32, 
                    shape=[FIRST_LAYER_SIZE,
                           SECOND_LAYER_SIZE])]*LAYER_REPETITIONS[1]
)
prune.append(
    [tf.placeholder(tf.float32, shape=[SECOND_LAYER_SIZE, 10])]
)

first_layers_per_second_layer = LAYER_REPETITIONS[0] // LAYER_REPETITIONS[1]

W = []
b = []
h_layer = []

W.append(
    [weight_variable([784, FIRST_LAYER_SIZE])]*LAYER_REPETITIONS[0]
)
b.append(
    [bias_variable([FIRST_LAYER_SIZE])] * LAYER_REPETITIONS[0]
)
h_layer.append(
    [tf.nn.relu(tf.matmul(x, W[0][i] * prune[0][i]) + b[0][i])
     for i in range(LAYER_REPETITIONS[0])]
)

W.append(
    [weight_variable([FIRST_LAYER_SIZE, SECOND_LAYER_SIZE])] *
    LAYER_REPETITIONS[1]
)
b.append(
    [bias_variable([SECOND_LAYER_SIZE])] * LAYER_REPETITIONS[1]
)
next_h_layer = []
for i in range(LAYER_REPETITIONS[1]):
    all_inputs = []
    for k in range(first_layers_per_second_layer):
        j = i * first_layers_per_second_layer + k
        all_inputs.append(h_layer[0][j])
    sum_input = tf.reduce_sum(all_inputs, axis=0)
    next_h_layer.append(tf.nn.relu(
        tf.matmul(sum_input, W[1][i] * prune[1][i]) + b[1][i]))
h_layer.append(next_h_layer)

W.append([weight_variable([SECOND_LAYER_SIZE, 10])])
b.append([bias_variable([10])])
y = tf.matmul(tf.reduce_sum(h_layer[-1], axis=0), W[2][0] * prune[2][0]) + b[2][0]

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
                **{p_ph: pr for p_list in zip(prune, p) for p_ph, pr in zip(*p_list)}
            })
            accuracies.append(acc)
            iterations.append(i)
            print("test accuracy in step {}: {}" .format(i, acc))
        train_step.run(feed_dict={
            x: batch[0],
            y_: batch[1],
            **{p_ph: pr for p_list in zip(prune, p) for p_ph, pr in zip(*p_list)}
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

    W_init = [[w.eval() for w in layer] for layer in W]

    p = [[np.ones([784, FIRST_LAYER_SIZE])]*LAYER_REPETITIONS[0],
         [np.ones([FIRST_LAYER_SIZE, SECOND_LAYER_SIZE])]*LAYER_REPETITIONS[1],
         [np.ones([SECOND_LAYER_SIZE, 10])]]

    base_accuracies, base_iterations = train(p)

    np.save('{}/base_accuracies_{}.npy'.format(
        RESULT_PATH_PREFIX, sample_n), base_accuracies)
    np.save('{}/base_iterations_{}.npy'.format(
        RESULT_PATH_PREFIX, sample_n), base_iterations)

    W_after_train = [w.eval() for w in W]

    for counts in REMAINING_COUNTS:
        if ITERATIVE:
            p = [create_prune(w.eval()*pr, c)
                 for layer in zip(W, p, counts)
                 for w, pr, c in zip(*layer)]
        else:
            p = [create_prune(w_after_train, c)
                 for layer in zip(W_after_train, counts)
                 for w_after_train, c in zip(*layer)]

        sess.run(tf.global_variables_initializer())
        for layer in zip(W, W_init):
            for w, w_init in zip(*layer):
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
