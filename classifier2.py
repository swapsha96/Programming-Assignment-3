import csv
import numpy as np
import heapq
from utils import *


if __name__ == "__main__":
    # Load complete speech data
    training_data = get_all_features_from_folder("Train")
    test_data = get_all_features_from_folder("Test")

    # initialization
    class_names = list(training_data.keys())
    confusion_matrix = np.zeros((len(class_names), len(class_names))).astype(np.int)

    

    # print("Confusion Matrix: ")
    # print(confusion_matrix)

    # print("Classification Accuracy: " + str(get_accuracy(confusion_matrix) * 100) + "%")

    # print("Class Precisions:")
    # precision = get_precision(confusion_matrix)
    # for name, p in zip(all_paths.keys(), precision):
    #     print("\t" + name + ": " + str(p * 100) + "%")
    # print("Mean Precision: " + str(get_mean_precision(confusion_matrix) * 100) + "%")

    # print("Class Recalls:")
    # recall = get_recall(confusion_matrix)
    # for name, r in zip(all_paths.keys(), recall):
    #     print("\t" + name + ": " + str(r * 100) + "%")
    # print("Mean Recall: " + str(get_mean_recall(confusion_matrix) * 100) + "%")

    # print("Class F-measures:")
    # f_measure = get_f_measure(get_precision(confusion_matrix), get_recall(confusion_matrix))
    # for name, f in zip(all_paths.keys(), f_measure):
    #     print("\t" + name + ": " + str(f * 100) + "%")
    # print("Mean F-measure: " + str(get_mean_f_measure(get_precision(confusion_matrix), get_recall(confusion_matrix)) * 100) + "%")
