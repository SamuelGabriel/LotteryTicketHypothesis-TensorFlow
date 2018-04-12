import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser("Arguments")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument("--resultdir", type=str)
parser.add_argument("--samplestart", type=int, default=0)
parser.add_argument("--samplesupremum", type=int, default=5)
parser.add_argument('--reductionsteps', type=int, default=10)
args = parser.parse_args()
assert(args.resultdir is not None)

START_SAMPLES_INDEX = args.samplestart
SUPREMUM_SAMPLES_INDEX = args.samplesupremum
NUM_REDUCE_STEPS = args.reducesteps
JUMP_REDUCE_STEP = 3
REDUCE_STEP_SIZE = 0.8 ** 2
RESULTS_DIR = args.resultdir

iterations = np.load(RESULTS_DIR + '/base_iterations_0.npy')
max_base_accuracies = np.zeros(iterations.shape)
min_base_accuracies = np.full(iterations.shape, np.inf)
avg_base_accuracies = np.zeros(iterations.shape)
for sample in range(START_SAMPLES_INDEX, SUPREMUM_SAMPLES_INDEX):
    accuracies = np.load(
        RESULTS_DIR + '/base_accuracies_{}.npy'.format(sample))
    max_base_accuracies = np.maximum(accuracies, max_base_accuracies)
    min_base_accuracies = np.minimum(accuracies, min_base_accuracies)
    avg_base_accuracies += accuracies
avg_base_accuracies = avg_base_accuracies/(sample+1)
plus_err = max_base_accuracies - avg_base_accuracies
minus_err = avg_base_accuracies - min_base_accuracies
plt.errorbar(iterations, avg_base_accuracies,
             yerr=np.stack([plus_err, minus_err], axis=0))

for share_it in range(0, NUM_REDUCE_STEPS, JUMP_REDUCE_STEP):
    max_accuracies = np.zeros(iterations.shape)
    min_accuracies = np.full(iterations.shape, np.inf)
    avg_accuracies = np.zeros(iterations.shape)
    for sample in range(START_SAMPLES_INDEX, SUPREMUM_SAMPLES_INDEX):
        accuracies = np.load(
            '{}/accuracies_sample_{}_restshare_step_{}.npy'
            .format(RESULTS_DIR, sample, share_it)
        )
        max_accuracies = np.maximum(accuracies, max_accuracies)
        min_accuracies = np.minimum(accuracies, min_accuracies)
        avg_accuracies += accuracies
    avg_accuracies = avg_accuracies/(sample+1)
    plus_err = max_accuracies - avg_accuracies
    minus_err = avg_accuracies - min_accuracies
    plt.errorbar(iterations, avg_accuracies,
                 yerr=np.stack([plus_err, minus_err], axis=0))

for share_it in range(0, NUM_REDUCE_STEPS, JUMP_REDUCE_STEP):
    max_accuracies = np.zeros(iterations.shape)
    min_accuracies = np.full(iterations.shape, np.inf)
    avg_accuracies = np.zeros(iterations.shape)
    for control_ex in range(3):
        for sample in range(START_SAMPLES_INDEX, SUPREMUM_SAMPLES_INDEX):
            accuracies = np.load(
                '{}/accuracies_sample_{}_control_{}_restshare_step_{}.npy'
                .format(RESULTS_DIR, sample, control_ex, share_it)
            )
            max_accuracies = np.maximum(accuracies, max_accuracies)
            min_accuracies = np.minimum(accuracies, min_accuracies)
            avg_accuracies += accuracies
    avg_accuracies = avg_accuracies/(sample + 1)/(control_ex + 1)
    plus_err = max_accuracies - avg_accuracies
    minus_err = avg_accuracies - min_accuracies
    print('yo')
    plt.errorbar(iterations, avg_accuracies,
                 yerr=np.stack([plus_err, minus_err], axis=0))

percent = ['{:0.2f}%'.format((REDUCE_STEP_SIZE ** (share_it)) * 100)
           for share_it in range(1, NUM_REDUCE_STEPS + 1, JUMP_REDUCE_STEP)]
plt.legend(['base']+percent+['control: ' +
                             p for p in percent], loc='lower right')
plt.show()
