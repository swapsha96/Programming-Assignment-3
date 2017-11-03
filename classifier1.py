import csv
import numpy as np
import heapq
from utils import *


def get_class_label(x, dataset, k):
    all_dtw_distances = []
    for class_name, all_features in dataset.items():
        for feature in all_features:
            d = -1 * dtw_distance(feature, x)
            if len(all_dtw_distances) < k:
                heapq.heappush(all_dtw_distances, (d, class_name))
            else:
                spilled_distance = heapq.heappushpop(all_dtw_distances, (d, class_name))
    
    bag = {}
    for (d, label) in all_dtw_distances:
        if label not in bag.keys():
            bag[label] = 0
        bag[label] += 1
    print("BAG", bag)
    label = max(bag, key=lambda k: bag[k])
    del bag
    del all_dtw_distances
    return label


if __name__ == "__main__":
    # Load complete speech data
    training_data = get_all_features_from_folder("dataset2/Train")
    test_data = get_all_features_from_folder("dataset2/Test")

    # initialization
    class_names = list(training_data.keys())
    confusion_matrix = np.zeros((len(class_names), len(class_names))).astype(np.int)
    k = 25

    counter, total = 0, 0
    for key, v in test_data.items():
        total += len(v)

    for test_class_name, all_test_features in test_data.items():
        for test_feature in all_test_features:
            class_label = get_class_label(test_feature, training_data, k)
            counter += 1
            print(class_label, test_class_name, (counter * 100) / total)
            confusion_matrix[class_i(class_names, test_class_name)][class_i(class_names, class_label)] += 1

    print("k: " + str(k))
    print("Confusion Matrix: ")
    print(confusion_matrix)

    print("Classification Accuracy: " + str(get_accuracy(confusion_matrix) * 100) + "%")

    print("Class Precisions:")
    precision = get_precision(confusion_matrix)
    for name, p in zip(class_names, precision):
        print("\t" + name + ": " + str(p * 100) + "%")
    print("Mean Precision: " + str(get_mean_precision(confusion_matrix) * 100) + "%")

    print("Class Recalls:")
    recall = get_recall(confusion_matrix)
    for name, r in zip(class_names, recall):
        print("\t" + name + ": " + str(r * 100) + "%")
    print("Mean Recall: " + str(get_mean_recall(confusion_matrix) * 100) + "%")

    print("Class F-measures:")
    f_measure = get_f_measure(get_precision(confusion_matrix), get_recall(confusion_matrix))
    for name, f in zip(class_names, f_measure):
        print("\t" + name + ": " + str(f * 100) + "%")
    print("Mean F-measure: " + str(get_mean_f_measure(get_precision(confusion_matrix), get_recall(confusion_matrix)) * 100) + "%")
