import svm
import numpy as np
from scipy.io import arff
import random
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

if __name__ == '__main__':
    filename = 'sonar.arff'
    data, meta = arff.loadarff(filename)
    X = []
    y = []
    for elem in data:
        X.append(list(elem)[:-1])
        y.append(elem[-1])
    X = np.array(X)
    y = np.array(y)
    # split the data into training and testing 80% to 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = svm.SVM(C=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    correct_rate = np.float64(len(np.nonzero(y_test == y_pred)[0])) / len(y_test)
    print 'correct_rate: %.2f' % correct_rate
    print 'predict\t true'
    for yp, yt in zip(y_pred, y_test):
        print '%s\t%s' % (yp, yt)

    # sklearn svm
    clf = SVC(C=10)
    clf.fit(X_train, y_train)
    y_pred_gs = clf.predict(X_test)
    correct_rate_gs = np.float64(len(np.nonzero(y_test == y_pred_gs)[0])) / len(y_test)
    print 'correct_rate (gold standard): %.2f' % correct_rate_gs