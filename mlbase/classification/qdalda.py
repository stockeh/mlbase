import sys  # for sys.float_info.epsilon

import numpy as np

######################################################################
# class QDA
######################################################################


class QDA(object):

    def __init__(self):
        # Define all instance variables here. Not necessary
        self.means = None
        self.stds = None
        self.mu = None
        self.sigma = None
        self.sigma_inv = None
        self.prior = None
        self.determinant = None
        self.discriminant_constant = None

    def train(self, X, T):
        self.classes = np.unique(T)
        self.means, self.stds = np.mean(X, 0), np.std(X, 0)
        Xs = (X - self.means) / self.stds
        self.mu = []
        self.sigma = []
        self.sigma_inv = []
        self.determinant = []
        self.prior = []
        nSamples = X.shape[0]
        for k in self.classes:
            rows_this_class = (T == k).reshape((-1))
            self.mu.append(np.mean(Xs[rows_this_class, :], 0).reshape((-1, 1)))
            self.sigma.append(np.cov(Xs[rows_this_class, :], rowvar=0))
            if self.sigma[-1].size == 1:
                self.sigma[-1] = self.sigma[-1].reshape((1, 1))
            det = np.linalg.det(self.sigma[-1])
            if det == 0:
                det = sys.float_info.epsilon
            self.determinant.append(det)
            # pinv in case Sigma is singular
            self.sigma_inv.append(np.linalg.pinv(self.sigma[-1]))
            self.prior.append(np.sum(rows_this_class) / float(nSamples))
        self._finish_train()

    def _finish_train(self):
        self.discriminant_constant = []
        for ki in range(len(self.classes)):
            self.discriminant_constant.append(
                np.log(self.prior[ki]) - 0.5 * np.log(self.determinant[ki])
            )

    def use(self, X, all_outputs=False):
        Xs = (X - self.means) / self.stds
        discriminants, probabilities = self._discriminant_function(Xs)
        predicted_class = self.classes[np.argmax(discriminants, axis=1)]
        predicted_class = predicted_class.reshape((-1, 1))
        return (
            (predicted_class, probabilities, discriminants)
            if all_outputs
            else predicted_class
        )

    def _discriminant_function(self, Xs):
        nSamples = Xs.shape[0]
        discriminants = np.zeros((nSamples, len(self.classes)))
        for ki in range(len(self.classes)):
            Xc = Xs - self.mu[ki].T
            discriminants[:, ki : ki + 1] = self.discriminant_constant[
                ki
            ] - 0.5 * np.sum(np.dot(Xc, self.sigma_inv[ki]) * Xc, axis=1).reshape(
                (-1, 1)
            )
        D = Xs.shape[1]
        probabilities = np.exp(discriminants - 0.5 * D * np.log(2 * np.pi))
        return discriminants, probabilities

    def __repr__(self):
        if self.mu is None:
            return "QDA not trained."
        else:
            return "QDA trained for classes {}".format(self.classes)


######################################################################
# class LDA
######################################################################


class LDA(QDA):

    def _finish_train(self):
        self.sigma_mean = np.sum(
            np.stack(self.sigma) * np.array(self.prior)[:, np.newaxis, np.newaxis],
            axis=0,
        )
        self.sigma_mean_inv = np.linalg.pinv(self.sigma_mean)
        # print(self.sigma)
        # print(self.sigma_mean)
        self.discriminant_constant = []
        self.discriminant_coefficient = []
        for ki in range(len(self.classes)):
            sigma_mu = np.dot(self.sigma_mean_inv, self.mu[ki])
            self.discriminant_constant.append(-0.5 * np.dot(self.mu[ki].T, sigma_mu))
            self.discriminant_coefficient.append(sigma_mu)

    def _discriminant_function(self, Xs):
        nSamples = Xs.shape[0]
        discriminants = np.zeros((nSamples, len(self.classes)))
        for ki in range(len(self.classes)):
            discriminants[:, ki : ki + 1] = self.discriminant_constant[ki] + np.dot(
                Xs, self.discriminant_coefficient[ki]
            )
        D = Xs.shape[1]
        probabilities = np.exp(
            discriminants
            - 0.5 * D * np.log(2 * np.pi)
            - 0.5 * np.log(self.determinant[ki])
            - 0.5
            * np.sum(np.dot(Xs, self.sigma_mean_inv) * Xs, axis=1).reshape((-1, 1))
        )
        return discriminants, probabilities


######################################################################
# Example use
######################################################################

if __name__ == "__main__":

    D = 5  # number of components in each sample
    N = 20  # number of samples in each class
    X = np.vstack(
        (np.random.normal(0.0, 1.0, (N, D)), np.random.normal(4.0, 1.5, (N, D)))
    )
    T = np.vstack(
        (np.array([1] * N).reshape((N, 1)), np.array([2] * N).reshape((N, 1)))
    )

    qda = QDA()
    qda.train(X, T)
    c, prob, _ = qda.use(X, all_outputs=True)
    print("QDA", np.sum(c == T) / X.shape[0] * 100, "% correct")
    print("{:>3s} {:>4s} {:>14s}".format("T", "Pred", "prob(C=k|x)"))
    for row in np.hstack((T, c, prob)):
        print("{:3.0f} {:3.0f} {:8.4f} {:8.4f}".format(*row))

    lda = LDA()
    lda.train(X, T)
    c, prob, d = lda.use(X, all_outputs=True)
    print("LDA", np.sum(c == T) / X.shape[0] * 100, "% correct")
    print("{:>3s} {:>4s} {:>14s}".format("T", "Pred", "prob(C=k|x)"))
    for row in np.hstack((T, c, prob)):
        print("{:3.0f} {:3.0f} {:8.4f} {:8.4f}".format(*row))
