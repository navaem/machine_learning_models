import decisiontree, gridsearchSVM, knn, logisticregression, supportvectormachine
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
        sklearn.utils.shuffle(x_data, y_data)
        train_data = x_data[20000:30000], y_data[20000:30000]
        test_data = y_data[30000:40000], y_data[30000:40000]
        print("Training Data 1 support: "
              + np.sum(train_data[1])/len(train_data[1]))
        print("Test Data 1 support: "
              + np.sum(train_data[0])/len(train_data[0]))

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
