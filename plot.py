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
parser.add_argument('--remainingcounts', type=str,
                    default='[(round(share*784*300), round(share*300*100), round(share*100*10)) ' +
                    'for share in ((.8**2)**i for i in range(1, 11))]')
args = parser.parse_args()
assert(args.resultdir is not None)

START_SAMPLES_INDEX = args.samplestart
SUPREMUM_SAMPLES_INDEX = args.samplesupremum
RESULTS_DIR = args.resultdir
REMAINING_COUNTS = eval(args.remainingcounts)

iterations = np.load(RESULTS_DIR + '/base_iterations_0.npy')


def plot_error(sample_accuracy_filenames):
    max_accuracies = np.zeros(iterations.shape)
    min_accuracies = np.full(iterations.shape, np.inf)
    avg_accuracies = np.zeros(iterations.shape)
    for i, sample_file in enumerate(sample_accuracy_filenames):
        accuracies = np.load(sample_file)
        max_accuracies = np.maximum(accuracies, max_accuracies)
        min_accuracies = np.minimum(accuracies, min_accuracies)
        avg_accuracies += accuracies
    avg_accuracies = avg_accuracies/(i+1)
    plus_err = max_accuracies - avg_accuracies
    minus_err = avg_accuracies - min_accuracies
    plt.errorbar(iterations, avg_accuracies,
                 yerr=np.stack([plus_err, minus_err], axis=0))


base_accuracies = [RESULTS_DIR + f'/base_accuracies_{sample}.npy'
                   for sample in range(START_SAMPLES_INDEX,
                                       SUPREMUM_SAMPLES_INDEX)]
plot_error(base_accuracies)
for remaining_count in REMAINING_COUNTS:
    accuracies = [RESULTS_DIR +
                  f'/accuracies_sample_{i}_restcounts_{remaining_count}.npy'
                  for i in range(START_SAMPLES_INDEX, SUPREMUM_SAMPLES_INDEX)]
    plot_error(accuracies)

plt.legend(['base']+REMAINING_COUNTS, loc='lower right')
plt.show()
