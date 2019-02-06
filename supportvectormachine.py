import numpy as np
from sklearn import svm
import argparse
import datasets.uci_adult


class SupportVectorClassifier:

    clf_model = None

    def __init__(self):
        self.clf_model = svm.LinearSVC()
        print("Created SupportVectorClassifier")

    def train(self, train_data):
        data = train_data[0]
        labels = train_data[1]
        print("Fitting model to training data.")
        self.clf_model.fit(data, labels)

    def test(self, data):
        print("Beginning Test")
        predict_vec = self.clf_model.predict(data[0])
        predict_vec = predict_vec == data[1]
        print("Test accuracy: ", np.sum(predict_vec)/len(predict_vec))


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

    model = SupportVectorClassifier()
    model.train(train_data)
    model.test(test_data)
