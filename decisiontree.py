import pickle
import numpy as np
from sklearn import tree
import argparse

class DecisionTree:

    clf_model = None

    def __init__(self):
        self.clf_model = tree.DecisionTreeClassifier()

    def train(self, train_data):
        data = train_data[0]
        labels = train_data[1]
        self.clf_model.fit(data, labels)

    def test(self, data):
        predict_vec = self.clf_model.predict(data[0])
        predict_vec = predict_vec == data[1]
        print("Test accuracy: ", np.sum(predict_vec))


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--train_data", default="")
    parse.add_argument("--test_data", default="")
    args = parse.parse_args()

    train_data = args.train_data
    test_data = args.test_data

    train_data = pickle.load(open(train_data, "rb"))
    test_data = pickle.load(open(test_data, "rb"))

    model = DecisionTree()
    model.train(train_data)
    model.test(test_data)
