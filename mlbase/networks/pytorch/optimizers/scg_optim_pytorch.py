import numpy as np
import math
import torch
import sys
from torch.optim import Optimizer

# def logging(s):
#     with open('newway.log', 'a') as f:
#         f.write(s)
#         f.write('\n')


class SCG(Optimizer):
    r"""Implements M. Møller's Scaled Conjugate Gradient Algorithm,
    based on
      "A Scaled Conjugate Gradient Algorithm for Fast Supervised
       Learning", by Martin Møller, Neural Networks, 6(4), 525-533, 1993,
    and
       the scg implementation in "NETLAB: Algorithms for Pattern
       Recognition", by Ian Nabney, Springer, 2002

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.scg(model.parameters())
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward(create_graph=True)   FIX THIS
        >>> optimizer.step()  FIX THIS
    """

    def __init__(self, parameters, minibatch=False, momentum=False):

        defaults = dict(
            sigma0=1e-6,
            lamb_min=1e-15,
            lamb_max=1e20,
            minibatch=minibatch,
            momentum=momentum)

        super().__init__(parameters, defaults)

    def _set_weights(self, w):
        torch.nn.utils.vector_to_parameters(w, self.param_groups[0]['params'])

    def _get_weights(self):
        return torch.nn.utils.parameters_to_vector(self.param_groups[0]['params'])

    def _gradients_to_vector(self):
        grads = []
        for param in self.param_groups[0]['params']:
            grads.append(param.grad.view(-1))
        # return negative gradient
        return -torch.cat(grads)

    def _loss_temp_w(self, w):
        old_weights = self._get_weights()
        self._set_weights(w)
        # logging('---------calling self.loss_closure grad=False  _loss_temp_w------------')
        loss = self.loss_closure()  # evaluate=True
        self._set_weights(old_weights)
        return self._gradients_to_vector(), loss

    def _gradient_temp_w(self, w):
        old_weights = self._get_weights().clone()
        self._set_weights(w)
        # logging('---------calling self.loss_closure grad=True   _gradient_temp_w------------')
        self.loss_closure()
        self._set_weights(old_weights)
        return self._gradients_to_vector()

    def _gradient(self):
        # logging('---------calling self.loss_closure no args  _gradient------------')
        loss = self.loss_closure()
        return self._gradients_to_vector(), loss

    @torch.no_grad()
    def step(self, closure=None):
        """Take one gradient descent step as defined by the Scaled Conjugate Gradient algorithm.

        Parameters
        ----------
        closure : function as defined here:

            Parameters
            ----------

            grad: boolean   default True

            loss: None (default) or precalculated loss value

                If loss is None, calculate loss.
                If grad is False, just return the loss, precalculated or just calculated.
                If grad is True, False and loss is None

                    False, only calculate and return the loss (forward pass)
                If grad is True, calculate gradient and return loss

            Example for minimizing Rosenbrock function:

                def closure(grad=True, loss=None):
                    if not loss:
                        loss = rosenbrock(xy_t)
                    if not grad:
                        return loss
                    optimizer.zero_grad()
                    loss.backward()
                    return loss

            Returns
            -------

            loss
        """

        # Make sure the closure is always called with grad enabled
        # torch.cuda.empty_cache()
        self.loss_closure = torch.enable_grad()(closure)

        sigma0 = self.defaults['sigma0']
        lamb_min = self.defaults['lamb_min']
        lamb_max = self.defaults['lamb_max']
        minibatch = self.defaults['minibatch']

        momentum = self.defaults['momentum']

        # 1) init
        momentum = 0.9
        first = False
        if not self.state:
            gw, fw = self._gradient()
            if momentum:
                gw = (1 - momentum) * gw
            p = gw  # conjugate direction
            p2 = p @ p  # .T
            self.state = {
                'lamb': 1e-6,
                'lamb_h': 0,
                'success': True,
                'n_successes': 0,
                'n_vars': len(self._get_weights()),
                'gw': gw,
                'fw': fw,
                'p': p,
                'p2': p2,
                'losses': [],
                'delta': 0,
            }
            first = True

        lamb = self.state['lamb']
        lamb_h = self.state['lamb_h']
        success = self.state['success']
        n_successes = self.state['n_successes']
        n_vars = self.state['n_vars']
        gw = self.state['gw']
        fw = self.state['fw']
        p = self.state['p']
        p2 = self.state['p2']
        losses = self.state['losses']
        delta = self.state['delta']

        if not first and momentum:
            gw_t, fw_t = self._gradient()
            gw = momentum * gw + (1 - momentum) * gw_t
            p = momentum * p + (1 - momentum) * gw_t
            p2 = p @ p  # .T

        w = self._get_weights()

        # 2) calculate second order info
        if success:
            sigma = sigma0 / math.sqrt(p2) # + ( ... np.finfo(float).eps)

            g_small_step = self._gradient_temp_w(w + sigma * p)

            s = (gw - g_small_step) / sigma
            delta = p @ s  # .T # Hessian matrix times a vector

            if p2 < sys.float_info.epsilon:
                return fw

        # 3) scale delta
        delta = delta + (lamb - lamb_h) * p2

        # 4) make Hessian positive definite
        if delta <= 0:
            lamb_h = 2 * (lamb - delta / p2)
            delta = - delta + lamb * p2
            lamb = lamb_h

        if delta == 0:
            print('===========================delta == 0  take care of this case')

        # 5) calculate step size
        mu = p @ gw  # .T
        alpha = mu / delta
        walpha = w + alpha * p
        g_temp, fwalpha = self._loss_temp_w(walpha)

        # 6) calculate comparison parameter
        Delta = 2 * delta * (fw - fwalpha) / mu**2

        # 7) update if successful reduction in error can be made
        if Delta >= 0:
            success = True
            lamb_h = 0
            n_successes += 1

            fw = fwalpha
            g_prev = gw.clone()
            self._set_weights(walpha)
            gw = g_temp

            # restart algorithm every len(w) iterations
            # print(n_successes, n_vars)
            if n_successes % n_vars == 0:
                p = gw
                n_successes = 0
            else:  # update search direction using Polak-Ribiere formula
                gamma = (gw @ gw - gw @ g_prev) / (g_prev @ g_prev)
                p = gw + gamma * p
            p2 = p @ p  # .T

            # reduce scale parameter
            if Delta >= 0.75:
                lamb = max(0.25 * lamb, lamb_min)
        else:
            success = False
            lamb_h = lamb

        # 8) increase scale parameter
        if Delta < 0.25 and p2 != 0:
            lamb = min(lamb + delta * (1 - Delta) / p2, lamb_max)

        losses.append(fw)

        self.state.update(
            {
                'lamb':  lamb,
                'lamb_h': lamb_h,
                'success': success,
                'n_successes': n_successes,
                'gw': gw,
                'fw': fw,
                'p': p,
                'p2': p2,
                'losses': losses,
                'delta': delta
            }
        )

        return fw


if __name__ == '__main__':
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    from torch.optim import Adam, SGD
    from tqdm import tqdm

    test_rosenbrock = True
    test_mnist = False

    ######################################################################
    if test_rosenbrock:
        # Example is adapted from mildlyoverfitted code and tutorial
        # https://github.com/jankrepl/mildlyoverfitted
        def rosenbrock(xy):
            x, y = xy
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        def run_optimization(xy_init, optimizer_class, n_iter, **optimizer_kwargs):
            xy_t = torch.tensor(xy_init, requires_grad=True)
            optimizer = optimizer_class([xy_t], **optimizer_kwargs)

            path = np.empty((n_iter + 1, 2))
            path[0, :] = xy_init

            def closure(grad=True, loss=None):
                '''Only for SCG'''
                if not loss:
                    loss = rosenbrock(xy_t)
                if not grad:
                    return loss
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
                return loss

            for i in tqdm(range(1, n_iter + 1)):

                if isinstance(optimizer, SCG):
                    optimizer.step(closure)

                else:
                    optimizer.zero_grad()
                    loss = rosenbrock(xy_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
                    optimizer.step()

                path[i, :] = xy_t.detach().numpy()

            return path

        def create_animation(paths,
                             colors,
                             names,
                             figsize=(12, 12),
                             x_lim=(-2, 2),
                             y_lim=(-1, 3),
                             n_seconds=5):
            if not (len(paths) == len(colors) == len(names)):
                raise ValueError

            path_length = max(len(path) for path in paths)

            n_points = 300
            x = np.linspace(*x_lim, n_points)
            y = np.linspace(*y_lim, n_points)
            X, Y = np.meshgrid(x, y)
            Z = rosenbrock([X, Y])

            minimum = (1.0, 1.0)

            fig, ax = plt.subplots(figsize=figsize)
            ax.contour(X, Y, Z, 90, cmap="jet")

            lines = [ax.plot([], [], '.-',
                             label=label,
                             c=c) for c, label in zip(colors, names)]

            ax.legend(prop={"size": 25})
            ax.plot(*minimum, "rD")

            def animate(i):
                for path, line in zip(paths, lines):
                    # set_offsets(path[:i, :])
                    line[0].set_xdata(path[:i+1, 0])
                    # set_offsets(path[:i, :])
                    line[0].set_ydata(path[:i+1, 1])

                ax.set_title(str(i))

            ms_per_frame = 1000 * n_seconds / path_length

            anim = FuncAnimation(
                fig, animate, frames=path_length, interval=ms_per_frame)
            return anim

        xy_init = (.3, .8)
        n_iter = 200

        path_adam = run_optimization(xy_init, Adam, n_iter, lr=0.01)
        path_sgd = run_optimization(xy_init, SGD, n_iter, lr=0.01)
        path_scg = run_optimization(xy_init, SCG, n_iter)

        freq = 1

        paths = [path_adam[::freq], path_sgd[::freq], path_scg[::freq]]
        colors = ["green", "blue", "black"]
        names = ["Adam", "SGD", "SCG"]

        anim = create_animation(paths,
                                colors,
                                names,
                                figsize=(12, 7),
                                x_lim=(-.1, 1.1),
                                y_lim=(-.1, 1.1),
                                n_seconds=7)

        print('sgd')
        print(path_sgd[-15:])
        print('adam')
        print(path_adam[-15:])
        print('scg')
        print(path_scg[-15:])

        print('Creating animation ...')
        anim.save("result.gif")

        print('Resulting animation is in result.gif')

    ######################################################################
    if test_mnist:
        print('='*70)
        print('MNIST')
        print('='*70)

        import pickle
        import gzip

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        print('Using device', device)

        with gzip.open('mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        n = train_set[0].shape[0]  # 1000
        n = 50000
        Xtrain = train_set[0][:n, :]
        Ttrain = train_set[1][:n]
        Xval = valid_set[0]
        Tval = valid_set[1]
        Xtest = test_set[0]
        Ttest = test_set[1]

        def to_torch_float(Z):
            return torch.from_numpy(Z.astype(np.float32)).to(device)

        def to_torch_int(Z):
            return torch.from_numpy(Z.astype(int)).to(device)

        Xtrain = to_torch_float(Xtrain)
        Ttrain = to_torch_int(Ttrain)
        Xval = to_torch_float(Xval)
        Tval = to_torch_int(Tval)
        Xtest = to_torch_float(Xtest)
        Ttest = to_torch_int(Ttest)

        print("Train with", Xtrain.shape[0], "images. Validate with",
              Xval.shape[0], "Test with", Xtest.shape[0], "images.")

        class NNet(torch.nn.Module):

            def __init__(self, n_inputs, hiddens, n_outputs):
                super().__init__()
                self.n_inputs = n_inputs
                self.layers = torch.nn.ModuleList()
                self.hiddens = hiddens
                ni = n_inputs
                for h in hiddens:
                    self.layers.append(torch.nn.Linear(ni, h))
                    self.layers.append(torch.nn.Tanh())
                    ni = h
                self.layers.append(torch.nn.Linear(ni, n_outputs))

            def get_device(self):
                return next(self.parameters()).device

            def forward(self, X):
                for layer in self.layers:
                    X = layer(X)
                return X

            def __repr__(self):
                return f'NNet({self.n_inputs}, {self.hiddens}, {self.n_outputs})'

        def get_batch(X, T, batch_size=100):
            if batch_size == -1:
                yield X, T
            else:
                n_samples = X.shape[0]
                rows = np.arange(n_samples)
                np.random.shuffle(rows)
                X = X[rows]
                T = T[rows]
                for first in range(0, n_samples, batch_size):
                    last = first + batch_size
                    # print(f'{first=} {last=}')
                    yield X[first:last], T[first:last]

        def percent_correct(Y_probs, T):
            predicted = Y_probs.argmax(-1)  # .cpu().numpy()
            return 100 * (predicted == T).sum() / T.numel()

        def likelihood(T, output):
            n = output.shape[0]
            return torch.mean(softmax(output, -1)[torch.arange(n), T])

        def softmax(Y, dim):
            maxY = torch.max(Y, dim=dim, keepdims=True)
            maxY = maxY[0]
            eY = torch.exp(Y - maxY)
            eY_sum = torch.sum(eY, dim=dim, keepdims=True)
            return eY / eY_sum

        # Small test problem

        # Xtrain = np.arange(100).reshape(-1, 1)
        # Ttrain = (Xtrain > 50).reshape(-1).astype(int)
        # Xtrain = to_torch_float(Xtrain)
        # Ttrain = to_torch_int(Ttrain)
        # Xval = Xtrain.clone()
        # Tval = Ttrain.clone()
        # Xtest = Xtrain.clone()
        # Ttest = Ttrain.clone()

        def run_mnist(batchsize):

            n_inputs = Xtrain.shape[1]
            hiddens = [1024]
            n_epochs = 1000
            n_outputs = 10
            nnet = NNet(n_inputs, hiddens, n_outputs)
            nnet.to(device)

            error_trace = []
            loss_func = torch.nn.CrossEntropyLoss()
            best_val_error = -np.inf

            batch_size = batchsize   # -1 for no minibatches
            if batch_size == -1 or batch_size == Xtrain.shape[0]:
                optimizer = SCG(nnet.parameters(), minibatch=False)
            else:
                optimizer = SCG(nnet.parameters(), minibatch=True)

            for epoch in range(n_epochs):

                for i, (Xb, Tb) in enumerate(
                        get_batch(Xtrain, Ttrain, batch_size)):

                    # Xb = Xb  # .to(device)
                    # Tb = Tb  # .to(device)

                    def closure(grad=True, loss=None):
                        """Only for SCG"""
                        if not loss:
                            outputs = nnet(Xb)
                            loss = loss_func(outputs, Tb)
                        if not grad:
                            return loss
                        optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(nnet.parameters(), 1.0)
                        return loss

                    optimizer.step(closure)

                    with torch.no_grad():
                        val = percent_correct(nnet(Xval), Tval).cpu()
                        if val > best_val_error:
                            # save weights
                            nnet.best_parameters = [p.clone()
                                                    for p in nnet.parameters()]
                            nnet.best_epoch = epoch + 1
                            best_val_error = val

                # nnet(Xtrain) ran out of GPU memory
                # error_trace.append([
                #                     likelihood(Tval, nnet(Xval)).detach().cpu(),
                #                     likelihood(Ttest, nnet(Xtest)).detach().cpu()])

                error_trace.append([
                    percent_correct(nnet(Xval), Tval).detach().cpu(),
                    percent_correct(nnet(Xtest), Ttest).detach().cpu()])

            error_trace = np.array(error_trace)
            print(error_trace[-10:, :])

            return error_trace

        print('Training with full batch, no minibatches')
        error_trace_nobatches = run_mnist(batchsize=-1)
        print('Training with minibatches of size 1000')
        error_trace_batches = run_mnist(batchsize=1000)

        plt.ion()  # for running in ipython

        plt.figure(5)
        plt.clf()
        plt.plot(error_trace_batches)
        plt.plot(error_trace_nobatches)
        plt.legend(('ValB', 'TestB', 'Val', 'Test'))
        plt.xlabel('Epochs')
        plt.ylabel('Precent Correct')
        # plt.legend(('TrainB', 'ValB', 'TestB', 'Train', 'Val', 'Test'))
        plt.show()
