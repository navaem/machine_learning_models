import numpy as np
from sklearn import model_selection
from sklearn import svm
import argparse
import datasets.uci_adult


class SupportVectorClassifier:

    clf_model = None

    def __init__(self):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        parameters = {'C': Cs, 'gamma' : gammas}
        svm_model = svm.SVC()
        self.clf_model = model_selection.GridSearchCV(svm_model, parameters)
        print("Created GridSearchCV")

    def train(self, t_data):
        data = t_data[0]
        labels = t_data[1]
        print("Fitting model to training data.")
        self.clf_model.fit(data, labels)
        print("Training Acc: ", self.clf_model.score(data, labels))

    def test(self, data):
        print("Beginning Test")
        predict_vec = self.clf_model.predict(data[0])
        predict_vec = predict_vec == data[1]
        test_acc = np.sum(predict_vec)/len(predict_vec)
        print("Test accuracy: ", test_acc)


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--train_data", default="")
    parse.add_argument("--test_data", default="")
    args = parse.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    if train_data == "":
        x_data, y_data = datasets.uci_adult.data()
        t_data = [x_data[0:10000], y_data[0:10000]]
        test_data = [x_data[10000:20000], y_data[10000:20000]]

    model = SupportVectorClassifier()
    model.train(train_data)
    model.test(test_data)
