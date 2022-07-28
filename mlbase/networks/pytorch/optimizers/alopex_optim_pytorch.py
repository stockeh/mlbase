import numpy as np
import torch
from torch.optim import Optimizer


class Alopex(Optimizer):
    r"""Implements ALgorithm Of Pattern EXtraction (Alopex),
    based on
      Unnikrishnan, K. P., and K. P. Venugopal. 1994. 
      "Alopex: A Correlation-Based Learning Algorithm for Feedforward and 
      Recurrent Neural Networks." Neural Computation 6 (3): 469–90.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        Example:
        >>> 
    """

    def __init__(self, parameters, eta0, N, decay=0.01):

        defaults = dict(
            eta0=eta0,
            N=N,
            decay=decay)

        super().__init__(parameters, defaults)

    def _set_weights(self, w):
        torch.nn.utils.vector_to_parameters(w, self.param_groups[0]['params'])

    def _get_weights(self):
        return torch.nn.utils.parameters_to_vector(self.param_groups[0]['params'])

    def _update_eta(self, epoch):
        self.state['eta'] = self.defaults['eta0'] / \
            (1. + self.defaults['decay'] * epoch)

    @torch.no_grad()
    def step(self, closure=None):
        """Take one setp as defined by Alopex
        """

        self.loss_closure = closure

        N = self.defaults['N']
        eta = self.defaults['eta0']

        w = self._get_weights()
        fw = self.loss_closure()

        # 1) init
        if not self.state:
            self.state = {
                't': 0.,
                'c_run': 0.,
                'p': torch.rand_like(w),
                'u': torch.empty_like(w),
                'losses': [],
                'term_reason': '',
                'iteration': 0,
            }

        t = self.state['t']
        c_run = self.state['c_run']
        p = self.state['p']
        u = self.state['u']
        losses = self.state['losses']
        term_reason = self.state['term_reason']
        iteration = self.state['iteration']

        # 2) update
        r = torch.rand_like(w)

        u[r < p] = -eta
        u[r >= p] = eta

        self._set_weights(w + u)

        dw = self._get_weights() - w
        de = self.loss_closure() - fw

        c = dw * de
        c_run = c_run + torch.abs(de)

        if (iteration % N) == 0:
            if iteration == 0:
                t = eta * c_run
            else:
                t = eta * c_run / N
                c_run = 0.

        p = 1. / (1. + torch.exp(-c / t))
        iteration += 1
        losses.append(fw)

        self.state.update(
            {
                't':  t,
                'c_run': c_run,
                'p': p,
                'u': u,
                'losses': losses,
                'term_reason': term_reason,
                'iteration': iteration
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

            def closure():
                '''Only for Alopex'''
                return rosenbrock(xy_t)

            for i in tqdm(range(1, n_iter + 1)):
                if isinstance(optimizer, Alopex):
                    loss = optimizer.step(closure)
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
        n_iter = 500

        path_adam = run_optimization(xy_init, Adam, n_iter, lr=0.01)
        path_sgd = run_optimization(xy_init, SGD, n_iter, lr=0.01)
        path_alopex = run_optimization(
            xy_init, Alopex, n_iter, eta0=0.002, N=25)

        freq = 1

        paths = [path_adam[::freq], path_sgd[::freq], path_alopex[::freq]]
        colors = ["green", "blue", "black"]
        names = ["Adam", "SGD", "Alopex"]

        print('sgd')
        print(path_sgd[-15:])
        print('adam')
        print(path_adam[-15:])
        print('alopex')
        print(path_alopex[-15:])

        anim = create_animation(paths,
                                colors,
                                names,
                                figsize=(12, 7),
                                x_lim=(-.1, 1.1),
                                y_lim=(-.1, 1.1),
                                n_seconds=7)

        print('Creating animation ...')
        anim.save("result_alopex.gif")
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
                optimizer = Alopex(nnet.parameters(), minibatch=False)
            else:
                optimizer = Alopex(nnet.parameters(), minibatch=True)

            for epoch in range(n_epochs):

                for i, (Xb, Tb) in enumerate(
                        get_batch(Xtrain, Ttrain, batch_size)):

                    # Xb = Xb  # .to(device)
                    # Tb = Tb  # .to(device)

                    def closure(grad=True, loss=None):
                        """Only for Alopex"""
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
