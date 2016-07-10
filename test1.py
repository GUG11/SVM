import numpy as np
import svm

# A simple test
if __name__ == '__main__':
    X = np.array([[0, 1, 0], [1, 0, 1]])
    y = np.array([-1, 1])
    clf = svm.SVM(C=10, kernel=('rbf', 1))
    clf.fit(X, y)
    pred = clf.predict(X)
    print pred