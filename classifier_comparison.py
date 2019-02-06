import decisiontree
import gridsearchSVM
import knn
import logisticregression
import supportvectormachine
import argparse
import datasets
import sklearn
import numpy as np


def main(arguments):
    dataset = arguments.use_dataset
    test_data = None
    train_data = None
    if dataset == "uci_adult":
        x_data, y_data = datasets.uci_adult.data(False)

        x_classify_one = x_data[y_data == 1]
        y_classify_one = y_data[y_data == 1]
        x_classify_zero = x_data[y_data == 0]
        y_classify_zero = y_data[y_data == 0]

        sklearn.utils.shuffle(x_classify_one, y_classify_one)
        sklearn.utils.shuffle(x_classify_zero, y_classify_zero)

        x_train = np.concatenate((x_classify_one[0:5000], x_classify_zero[0:5000]))
        y_train = np.concatenate((np.ones(5000), np.zeros(5000)))
        x_test = np.concatenate((x_classify_one[5000:10000], x_classify_zero[5000:10000]),)
        y_test = np.concatenate((np.ones(5000), np.zeros(5000)))
        sklearn.utils.shuffle(x_train, y_train)
        sklearn.utils.shuffle(x_test, y_test)

        train_data = [x_train, y_train]
        test_data = [x_test, y_test]
        print("Training Data 1 support: {}".format
              (np.sum(train_data[1])/len(train_data[1])))
        print("Test Data 1 support: {}".format
              (np.sum(test_data[1])/len(test_data[1])))

    print("-------------------------------"
          "Decision Tree"
          "-------------------------------")

    model = decisiontree.DecisionTree()
    model.train(train_data)
    model.test(test_data)

    print("--------------------------"
          "Logistic  Regression"
          "--------------------------")

    model = logisticregression.LogisticRegressionModel()
    model.train(train_data)
    model.test(test_data)

    print("------------------------------"
          "kNN  Model"
          "------------------------------")

    model = knn.KNeighborsClassifier()
    model.train(train_data)
    model.test(test_data)

    print("------------------------------"
          "SVM  Model"
          "------------------------------")

    model = supportvectormachine.SupportVectorClassifier()
    model.train(train_data)
    model.test(test_data)

    print("------------------------"
          "SVM  Model Grid Search"
          "------------------------")

    gridsearchSVM.SupportVectorClassifier()
    model.train(train_data)
    model.test(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dataset", default="")
    args = parser.parse_args()

    main(args)
