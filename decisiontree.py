import numpy as np
from sklearn import tree
import argparse
import datasets.uci_adult


class DecisionTree:

    clf_model: tree.DecisionTreeClassifier = None

    def __init__(self):
        self.clf_model = tree.DecisionTreeClassifier()
        print("Created DecisionTreeClassifier")

    def train(self, t_data):
        data = t_data[0]
        labels = t_data[1]
        print("Fitting model to training data.")
        self.clf_model.fit(data, labels)
        print("Training Acc: ", self.clf_model.score(data, labels))

    def test(self, data):
        print("Beginning Test")
        test_score = self.clf_model.score(data[0], data[1])
        print("Test accuracy: ", test_score)


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--train_data", default="")
    parse.add_argument("--test_data", default="")
    args = parse.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    if train_data == "":
        x_data, y_data = datasets.uci_adult.data()
        train_data = [x_data[0:10000], y_data[0:10000]]
        test_data = [x_data[10000:20000], y_data[10000:20000]]

    model = DecisionTree()
    model.train(train_data)
    model.test(test_data)
