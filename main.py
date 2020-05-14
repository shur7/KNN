import pandas as pd
import numpy as np
import math
import operator

TRAIN_FILE_PATH = './data/train.csv'
TEST_FILE_PATH = './data/test.csv'
DIRECTION_FILE_PATH = './data/trainDirection.csv'
FINAL_TEST_FILE_k1 = './results/testingk1.csv'
FINAL_TEST_FILE_k3 = './results/testingk3.csv'
FINAL_TEST_FILE_k5 = './results/testingk5.csv'
FINAL_TEST_FILE_k10 = './results/testingk10.csv'

train_data_frame = []
test_data_frame = []

def loadData(filePath):
    dataFrame = pd.read_csv(filePath)
    return dataFrame


def processData(df):
    directions_df = pd.read_csv(DIRECTION_FILE_PATH)
    df['Direction'] = directions_df['Direction']
    # print(df.head())
    return df

def transformTrainingData(train_data_frame):
    trainings = []
    for index, row in train_data_frame.iterrows():
        test_row = [round(row[0], 3), round(row[1], 3), row[2]]
        trainings.append(test_row)
    return trainings


def euclidianDistance(sample1, sample2, length):
    distance = 0
    for x in range(length):
        distance += pow(sample1[x] - sample2[x], 2)
    return math.sqrt(distance)


def getNeighbors(train_set, test_sample, k):
    distances = []
    neighbors = []
    length = len(test_sample - 1)

    for x in range(len(train_set)):
        dist = euclidianDistance(test_sample, train_set[x], length)
        distances.append((train_set[x], dist))

    distances.sort(key=operator.itemgetter(1))

    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return round((correct / float(len(testSet))) * 100.0, 3)

def knn(k, path, test_data, train_data, actual_test_data):

    print('*-*-*-*-*-*-*-*-*    K = ', k, '  *-*-*-*-*-*-*-*-*')
    copy_of_test_data = test_data.copy()
    copy_of_train_data = train_data
    predictions = []

    print('--> Calculating KNN...')
    for index, test_sample in test_data.iterrows():
        # print(test_sample.values[0], ', ', test_sample.values[1])
        neighbors = getNeighbors(copy_of_train_data, test_sample, k)
        neighbors_prediction = getResponse(neighbors)
        predictions.append(neighbors_prediction)

    print('--> Writing predictions...')
    copy_of_test_data['Direction'] = predictions
    copy_of_test_data.to_csv(path, float_format='%.3f')
    print('--> Calculating Accuracy...')
    accuracy = getAccuracy(actual_test_data, copy_of_test_data['Direction'].values)
    print('Accuracy = ', accuracy, '%')


train_data_frame = processData(loadData(TRAIN_FILE_PATH))
training_data = transformTrainingData(train_data_frame)
test_data_frame = loadData(TEST_FILE_PATH)
actual_test_result = pd.read_csv('./data/testing.csv')['Direction'].values


print('Training data: ', train_data_frame.shape[0], 'samples.')
print('Testing data: ', test_data_frame.shape[0], 'samples.')

knn(1, FINAL_TEST_FILE_k1, test_data_frame, training_data, actual_test_result)
knn(3, FINAL_TEST_FILE_k3, test_data_frame, training_data, actual_test_result)
knn(5, FINAL_TEST_FILE_k5, test_data_frame, training_data, actual_test_result)
knn(10, FINAL_TEST_FILE_k10, test_data_frame, training_data, actual_test_result)
