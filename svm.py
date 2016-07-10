import numpy as np

class SVM:
    """
        min 1/2 ||w||^2 + C \sum_{i=1}^N xi_i
        s.t. y_i(w^T x_i - b) >= 1 - xi_i
             xi >= 0
    """
    def __init__(self, C, kernel=('linear',)):
        """
        initialize SVM
        :param C: cost of misclassification
        :param kernel: kernel ('linear',), ('polynomial', degree),
        ('rbf', beta)
        """
        self._C = C
        self._eps = 1e-8
        self._kernel_type = kernel[0]
        if self._kernel_type == 'linear':
            self._K = lambda x, y: np.dot(x, y)
        elif self._kernel_type == 'polynomial':
            self._degree = kernel[1]
            self._K = lambda x, y: (np.dot(x, y) + 1)**self._degree
        elif self._kernel_type == 'rbf':
            self._beta = kernel[1]
            self._K = lambda x, y: np.exp(-self._beta * np.sum(np.square(x-y)))
        else:
            raise Exception('Wrong kernel type!')

    def fit(self, X, y, max_iter=1000, epsilon=1e-3):
        """
        fit SVM with data X = [x_1,x_2,...,x_n] and labels y = [y_1,...,y_n]
        The sequential minimal optimization algorithm is used.
        reference:
            John C. Platt. Sequential Minimal Optimization: A Fast Algorithm for
            Training Support Vector Machines. 1998
        :param X: data N x p
        :param y: data N x 1
        :param max_iter: maximum number of iterations
        :param epsilon: tolerance for y_i(w^T x_i - b) >= 1 - xi_i - epsilon
        :return: nothing
        """
        # initialize
        self._N, self._p = X.shape
        self._tol = epsilon
        self._X = X
        self.build_label_dict(y)
        self._w, self._b, self._a = np.zeros(self._p), 0, np.zeros(self._N)
        self._error_cache = dict()
        examine_all = 1
        for i in xrange(max_iter):
            print 'epoch %d' % i
            num_changed = 0
            if examine_all:
                for i in range(self._N):
                    num_changed += self.examineKKT(i)
            else:
                for i in range(self._N):
                    if self._a[i] > self._eps:
                        num_changed += self.examineKKT(i)
            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1
            if num_changed == 0 and not examine_all:
                break

    def examineKKT(self, i2):
        """
        check whether a sample violates KKT conditions
        a_i = 0 <=> y[i]u[i] >= 1
        0 < a_i < C <=> y[i]u[i] = 1
        a_i = C <=> y[i]u[i] <= 1
        :param i2: index of the sample
        :return: boolean:
        """
        x2 = self._X[i2]
        y2 = self._y[i2]
        a2 = self._a[i2]
        if i2 in self._error_cache:
            E2 = self._error_cache[i2]
        else:
            E2 = self.output(x2) - y2
        err2 = E2 * y2
        if (err2 < -self._tol and a2 < self._C) or \
            (err2 > self._tol and a2 > 0):
            # find 0 < a < C
            bounded_alpha_index = np.where((0 < self._a) & (self._a < self._C))[0]
            if 1 < len(bounded_alpha_index):
                # choose i1 maximize |E1 - E2|
                E1 = []
                for i in bounded_alpha_index:
                    if i in self._error_cache:
                        E1.append(self._error_cache[i])
                    else:
                        E1.append(self.output(self._X[i]))
                i1 = bounded_alpha_index[np.argmax(np.abs(E1-E2))]
                # first try update selected i1
                if self.update_multipliers(i1, i2):
                    return 1
                # if failed, loop over all non-zero and non-C alpha, starting at a random point
                start = np.random.randint(0, len(bounded_alpha_index))
                for i1 in range(start, len(bounded_alpha_index)):
                    if self.update_multipliers(i2, bounded_alpha_index[i1]):
                        return 1
                for i1 in range(0, start):
                    if self.update_multipliers(i2, bounded_alpha_index[i1]):
                        return 1
            # if failed, loop over all possible i1
            boundary_alpha_index = np.setdiff1d(np.arange(self._N), bounded_alpha_index)
            start = np.random.randint(0, len(boundary_alpha_index))
            for i1 in range(start, len(boundary_alpha_index)):
                if self.update_multipliers(i2, boundary_alpha_index[i1]):
                    return 1
            for i1 in range(0, start):
                if self.update_multipliers(i2, boundary_alpha_index[i1]):
                    return 1
        return 0

    def update_multipliers(self, i1, i2):
        """
        update two Lagragian multipliers
        :param i1: index of a1
        :param i2: index of a2
        :return: 1 (success) | 0 (success)
        """
        if i1 == i2:
            return 0
        a1, a2 = self._a[i1], self._a[i2]
        y1, y2 = self._y[i1], self._y[i2]
        x1, x2 = self._X[i1], self._X[i2]
        E1 = self._error_cache[i1] if i1 in self._error_cache else \
            self.output(x1) - y1
        E2 = self._error_cache[i2] if i2 in self._error_cache else \
            self.output(x2) - y2
        s = y1 * y2
        # compute L and H
        if s < 0:
            L = max(0, a2 - a1)
            H = min(self._C, self._C + a2 - a1)
        else:
            L = max(0, a1 + a2 - self._C)
            H = min(self._C, a1 + a2)
        if L == H:
            return 0
        K11 = self._K(x1, x1)
        K12 = self._K(x1, x2)
        K22 = self._K(x2, x2)
        eta = K11 + K22 - 2 * K12
        if eta > 0:
            a2_new = a2 + y2 * (E1-E2) / eta
            a2_clipped = L if a2_new < L else a2_new
            a2_clipped = H if a2_clipped > H else a2_clipped
        else:
            # compute the object function at a2=L and a2=H
            f1 = y1 * (E1 + self._b) - a1 * K11 - s * a2 * K12
            f2 = y2 * (E2 + self._b) - s * a1 * K12 - a2 * K22
            L1 = a1 + s * (a2 - L)
            H1 = a1 + s * (a2 - H)
            Psi_L = L1 * f1 + L * f2 + 0.5 * L1 * L1 * K11 + 0.5 * L * L \
                    * K22 + s * L * L1 * K12
            Psi_H = H1 * f1 + H * f2 + 0.5 * H1 * H1 * K11 + 0.5 * H * H \
                    * K22 + s * H * H1 * K12
            if Psi_L < Psi_H - self._eps:
                a2_clipped = L
            elif Psi_L > Psi_H + self._eps:
                a2_clipped = H
            else:
                a2_clipped = a2

        if abs(a2_clipped - a2) < self._eps * (a2 + a2_clipped + self._eps):
            return 0
        a1_new = a1 + s * (a2 - a2_clipped)
        # update threshold b
        b1 = E1 + y1 * (a1_new - a1) * K11 + y2 * (a2_clipped - a2) * K12 + self._b
        b2 = E2 + y1 * (a1_new - a1) * K12 + y2 * (a2_clipped - a2) * K22 + self._b
        self._b = (b1 + b2) * 0.5
        # update weight vector if SVM is linear
        if self._kernel_type == 'linear':
            self._w += y1 * (a1_new - a1) * x1 + y2 * (a2_clipped - a2) * x2
        self._a[i1] = a1_new
        self._a[i2] = a2_clipped
        return 1

    def output(self, x):
        """
            u = \sum_{i:a[i]>0} a[i]y[i]K(xi,x)
        :param x: data of one sample
        :return: u
        """
        if self._kernel_type == 'linear':
            return np.dot(self._w, x) - self._b
        else:
            u = -self._b
            nnz_indices = np.where(self._a > self._eps)[0]
            for i in nnz_indices:
                u += self._a[i] * self._y[i] * self._K(self._X[i], x)
            return u

    def predict(self, X):
        """
        predict labels of X
        :param X: n x p data set
        :return: n x 1 labels
        """
        if np.ndim(X) == 1:
            return self._y_dict[(self.output(X) >= 0) * 2 - 1]
        elif np.ndim(X) == 2:
            return [self._y_dict[(self.output(X[i]) >= 0) * 2 - 1]  \
                    for i in range(X.shape[0])]
        else:
            raise Exception('Invalid X shape')

    def build_label_dict(self, y):
        """
        build the label dictionary
        :param y: n x 1 labels
        :return: nothing
        """
        unique_label = np.unique(y)
        self._y_dict = {-1: unique_label[0], 1: unique_label[1]}
        self._y = np.where(y==unique_label[0], -1, 1)