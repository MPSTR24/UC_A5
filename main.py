import os
from sktime.datasets import load_from_tsfile
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier

def ten_cross_validation(clf, X, y):

    y_true = []
    y_pred = []

    """ create a 10 fold cross validation """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=44)

    # loop over the splits and fit the classifier for each one
    for train_idx, test_idx in cv.split(X, y):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # slice X along the first axis
        y_train, y_test = y[train_idx], y[test_idx]  # slice y

        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)

        y_true.extend(y_test)
        y_pred.extend(prediction)

    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)

    cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)

    # plot and save the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 9))
    cmd.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.xticks(rotation=45)
    plt.title("10 fold cross validation")
    plt.savefig('./confusion_matrix.png')
    print(accuracy_score(y_true, y_pred))


if __name__ == "__main__":
    
    CURRENT_PATH = os.getcwd()
    DATASET_PATH = os.path.join(CURRENT_PATH, "dataset", "saftey_recognition", "saftey_recognition.ts")

    X, y = load_from_tsfile(DATASET_PATH)

    print(np.shape(X))
    # print(y)

    # knn_classifier = KNeighborsTimeSeriesClassifier(distance='dtw')
    # ten_cross_validation(knn_classifier, X, y)
    
    rocket_classifier = RocketClassifier(num_kernels=1000)
    ten_cross_validation(rocket_classifier, X, y)