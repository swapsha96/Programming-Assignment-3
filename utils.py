import csv
import numpy as np
from os import walk


def get_accuracy(confusion_matrix):
    num, denom = 0, 0
    for diagonal_index in range(len(confusion_matrix)):
        num += confusion_matrix[diagonal_index][diagonal_index]

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            denom += confusion_matrix[i][j]

    return (num / denom)

def get_precision(confusion_matrix):
    precision = []
    for i, row in zip(range(len(confusion_matrix)), confusion_matrix):
        precision.append(confusion_matrix[i][i] / sum(row))
    return precision

def get_mean_precision(confusion_matrix):
    precision = get_precision(confusion_matrix)
    return (sum(precision) / len(precision))

def get_recall(confusion_matrix):
    recall = []
    for i, row in zip(range(len(confusion_matrix)), np.array(confusion_matrix).T):       
        try:
            recall.append(confusion_matrix[i][i] / sum(row))
        except (ZeroDivisionError, TypeError):
            recall.append('-')
    return recall

def get_mean_recall(confusion_matrix):
    recall = get_recall(confusion_matrix)
    try:
        return (sum(recall) / len(recall))
    except (ZeroDivisionError, TypeError):
        return '-'

def get_f_measure(precision, recall):
    measure = []
    for p, r in zip(precision, recall):
        try:
            measure.append((2 * p * r) / (p + r))
        except (ZeroDivisionError, TypeError):
            measure.append('-')
    return measure

def get_mean_f_measure(precision, recall):
    f_measure = get_f_measure(precision, recall)
    try:
        return (sum(f_measure) / len(f_measure))
    except (ZeroDivisionError, TypeError):
        return '-'

def load_csv_from_file(folder, file):
    with open(folder + "/" + file) as f:
        return np.array([line.strip().split(' ') for line in f]).astype(np.float)

def get_all_features_from_folder(folder):
    all_paths = {}
    for (dirpath, dirnames, filenames) in walk(folder):
        key = dirpath.split("/")[-1]
        if dirnames == []:
            for name in filenames:
                filetype = name.split(".")[-1]
                if filetype != "csv" and filetype != "mfcc":
                    continue
                if key not in all_paths.keys():
                    all_paths[key] = []
                all_paths[key].append(load_csv_from_file(dirpath, name))
    return all_paths

def min_neighbour_distance(matrix, i, j):
    bucket = []
    if (i-1) != -1 and j != -1:
        bucket.append(matrix[i-1][j])
    if i != -1 and (j-1) != -1:
        bucket.append(matrix[i][j-1])
    if (i-1) != -1 and (j-1) != -1:
        bucket.append(matrix[i-1][j-1])
    if len(bucket) == 0:
        return 0
    else:
        return min(bucket)

def min_neighbour_distance_i(matrix, i, j):
    bucket = []
    if (i-1) != -1 and j != -1:
        bucket.append((matrix[i-1][j], i-1, j))
    if i != -1 and (j-1) != -1:
        bucket.append((matrix[i][j-1], i, j-1))
    if (i-1) != -1 and (j-1) != -1:
        bucket.append((matrix[i-1][j-1], i-1, j-1))
    
    return min(bucket, key = lambda t: t[0])

def dtw_distance(list1, list2):
    matrix = np.zeros((len(list1), len(list2))).astype(np.float)
    for i in range(len(list1)):
        for j in range(len(list2)):
            matrix[i][j] = np.linalg.norm(list1[i] - list2[j]) + min_neighbour_distance(matrix, i, j)
    dist, i, j = matrix[-1][-1], len(list1) - 1, len(list2) - 1

    while i != 0 and j != 0:
        d, i, j = min_neighbour_distance_i(matrix, i, j)
        dist += d
    return dist


def class_i(keys, key):
    return list(keys).index(key)

if __name__ == "__main__":
    list1 = np.array([2, 3, 2, 1, 3, 4])
    list2 = np.array([1, 2, 5, 4, 3, 7])
    print(dtw_distance(list1, list2))
