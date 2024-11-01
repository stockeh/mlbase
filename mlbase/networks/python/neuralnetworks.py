import copy

import numpy as np

from mlbase.networks.python import optimizers as opt
from mlbase.utilities import mlutilities as ml


class NeuralNetwork:

    def __init__(self, n_inputs, n_hiddens_list, n_outputs, activation='tanh'):

        if not isinstance(n_hiddens_list, list):
            raise Exception(
                f'{type(self).__name__}: n_hiddens_list must be a list.')

        if n_hiddens_list == 0 or n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)

        self.n_inputs = n_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_outputs = n_outputs

        # Do we have any hidden layers?
        self.Vs = []
        ni = n_inputs
        for layeri in range(self.n_hidden_layers):
            n_in_layer = self.n_hiddens_list[layeri]
            self.Vs.append(1 / np.sqrt(1 + ni) *
                           np.random.uniform(-1, 1, size=(1 + ni, n_in_layer)))
            ni = n_in_layer
        self.W = 1/np.sqrt(1 + ni) * np.random.uniform(-1,
                                                       1, size=(1 + ni, n_outputs))

        if activation == 'relu':
            self.activation = lambda weighted_sum: weighted_sum * \
                (weighted_sum > 0)
            self.d_activation = lambda activation_val: 1. * \
                (activation_val > 0)

        elif activation == 'logistic':
            self.activation = lambda weighted_sum: 1. / \
                (1. + np.exp(-weighted_sum))
            self.d_activation = lambda activation_val: activation_val * \
                (1. - activation_val)

        elif activation == 'tanh':
            self.activation = lambda weighted_sum: np.tanh(weighted_sum)
            self.d_activation = lambda activation_val: 1 - activation_val**2

        else:
            raise Exception(
                f'{type(self).__name__}: activation must be either [tanh|relu|logistic].')

        # Member variables for standardization
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.trained = False
        self.reason = None
        self.error_trace = None
        self.n_epochs = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {
            self.n_hiddens_list}, {self.n_outputs})'
        if self.trained:
            str += f'\n   Network was trained for {self.n_epochs} epochs'
            str += f' that took {self.training_time:.4f} seconds. Final objective value is {
                self.error_trace[-1]:.3f}'
        else:
            str += '  Network is not trained.'
        return str

    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result

    def _unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans

    def _pack(self, Vs, W):
        return np.hstack([V.flat for V in Vs] + [W.flat])

    def _unpack(self, w):
        first = 0
        n_this_layer = self.n_inputs
        for i in range(self.n_hidden_layers):
            self.Vs[i][:] = w[first:first + (1 + n_this_layer) *
                              self.n_hiddens_list[i]].reshape((1 + n_this_layer, self.n_hiddens_list[i]))
            first += (1 + n_this_layer) * self.n_hiddens_list[i]
            n_this_layer = self.n_hiddens_list[i]
        self.W[:] = w[first:].reshape((1 + n_this_layer, self.n_outputs))

    def _forward_pass(self, X):
        # Assume weights already unpacked
        Z_prev = X  # output of previous layer
        Z = [Z_prev]
        for i in range(self.n_hidden_layers):
            V = self.Vs[i]
            Z_prev = self.activation(Z_prev @ V[1:, :] + V[0:1, :])
            Z.append(Z_prev)
        Y = Z_prev @ self.W[1:, :] + self.W[0:1, :]
        return Y, Z

    def _objectiveF(self, w, X, T):
        self._unpack(w)
        Y, _ = self._forward_pass(X)
        return 0.5 * np.mean((T - Y)**2)

    def _gradientF(self, w, X, T):
        self._unpack(w)
        Y, Z = self._forward_pass(X)
        # Do backward pass, starting with delta in output layer
        delta = -(T - Y) / (X.shape[0] * T.shape[1])
        # Another way to define dEdW without calling np.insert
        dW = np.vstack((np.sum(delta, axis=0), Z[-1].T @ delta))
        dVs = []
        delta = self.d_activation(Z[-1]) * (delta @ self.W[1:, :].T)
        for Zi in range(self.n_hidden_layers, 0, -1):
            Vi = Zi - 1  # because X is first element of Z
            dV = np.vstack((np.sum(delta, axis=0), Z[Zi-1].T @ delta))
            dVs.insert(0, dV)  # like append, but at front of list of dVs
            delta = (delta @ self.Vs[Vi][1:, :].T) * self.d_activation(Z[Zi-1])
        return self._pack(dVs, dW)

    def _setup_standardize(self, X, T):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy.copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1

    def _objective_to_actual(self, objective):
        return np.sqrt(objective)

    def train(self, X, T, n_epochs, method='scg',
              verbose=False, save_weights_history=False,
              learning_rate=0.001, momentum_rate=0.0):  # only for sgd and adam

        if X.shape[1] != self.n_inputs:
            raise Exception(
                f'train: number of columns in X ({X.shape[1]}) not equal to number of network inputs ({self.n_inputs})')

        self._setup_standardize(X, T)
        X = self._standardizeX(X)
        T = self._standardizeT(T)

        try:
            algo = [opt.sgd, opt.adam, opt.scg][[
                'sgd', 'adam', 'scg'].index(method)]
        except:
            raise Exception(
                "train: method={method} not one of 'scg', 'sgd' or 'adam'")
        result = algo(self._pack(self.Vs, self.W),
                      self._objectiveF,
                      [X, T], n_epochs,
                      self._gradientF,  # not used if scg
                      eval_f=self._objective_to_actual,
                      learning_rate=learning_rate, momentum_rate=momentum_rate,
                      verbose=verbose,
                      save_wtrace=save_weights_history)

        self._unpack(result['w'])
        self.reason = result['reason']
        # * self.Tstds  # to _unstandardize the MSEs
        self.error_trace = result['ftrace']
        self.n_epochs = len(self.error_trace) - 1
        self.trained = True
        self.weight_history = result['wtrace'] if save_weights_history else None
        self.training_time = result['time']
        return self

    def use(self, X, all_outputs=False):
        X = self._standardizeX(X)
        Y, Z = self._forward_pass(X)
        Y = self._unstandardizeT(Y)
        return (Y, Z[1:]) if all_outputs else Y

    def get_n_epochs(self):
        return self.n_epochs

    def get_error_trace(self):
        return self.error_trace

    def get_training_time(self):
        return self.training_time

    def get_weight_history(self):
        return self.weight_history

    def draw(self, input_names=None, output_names=None, gray=False):
        ml.draw(self.Vs, self.W, input_names, output_names, gray)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    np.random.seed(42)

    n = 500
    X = np.linspace(0, 2 * np.pi, n).reshape((-1, 1))
    T = np.sin(X) + 0.1 * np.random.randn(*X.shape)
    n_epochs = 2000

    nnet = NeuralNetwork(X.shape[1], [], T.shape[1])
    nnet.train(X, T, n_epochs)
    Y = nnet.use(X)
    print(
        f'scg  {nnet.n_hiddens_list} RMSE {ml.rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')

    plt.figure(1, figsize=(5, 4))

    nnet = NeuralNetwork(X.shape[1], [5]*2, T.shape[1])
    nnet.train(X, T, n_epochs)
    Y = nnet.use(X)
    print(
        f'scg  {nnet.n_hiddens_list} RMSE {ml.rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')

    plt.plot(nnet.error_trace, lw=2, label='scg')

    nnet = NeuralNetwork(X.shape[1], [5]*2, T.shape[1])
    nnet.train(X, T, n_epochs, method='sgd',
               learning_rate=0.01, momentum_rate=0.5)
    Y = nnet.use(X)
    print(
        f'sgd  {nnet.n_hiddens_list} RMSE {ml.rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')

    plt.plot(nnet.error_trace, lw=2, label='sgd')

    nnet = NeuralNetwork(X.shape[1], [5]*2, T.shape[1])
    nnet.train(X, T, n_epochs, method='adam', learning_rate=0.001)
    Y = nnet.use(X)
    print(
        f'adam {nnet.n_hiddens_list} RMSE {ml.rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')

    plt.plot(nnet.error_trace, lw=2, label='adam')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')

    plt.figure(2)
    nnet.draw()

    plt.show(block=True)
