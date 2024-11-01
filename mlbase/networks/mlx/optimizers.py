# MLX implemention by Jason Stock (2024)
#
# Based on the Scaled Conjugate Gradient algorithm by Møller (1993)

import copy
import sys
import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_flatten
from typing import List


def parameters_to_vector(parameters: dict):
    vec = []
    for _, val in tree_flatten(parameters):
        vec.append(val.flatten())
    return mx.concatenate(vec)


def vector_to_parameters(vec: mx.array, parameters: dict):
    i = 0
    for _, val in tree_flatten(parameters):
        n_param = val.size
        val[:] = vec[i:i+n_param].reshape(val.shape)
        i += n_param
    return parameters


class SCG():
    def __init__(
        self, loss_fn, grad_fn
    ):
        super().__init__()
        self._loss_fn = loss_fn
        self._grad_fn = grad_fn

        self._initialized = False
        self._state = {'step': mx.array(0, mx.uint64)}

        self.defaults = dict(
            sigma0=1e-6,
            lamb_min=1e-15,
            lamb_max=1e20,
        )

    @property
    def state(self):
        return self._state

    @property
    def step(self):
        return self.state['step']

    def init(self, parameters: dict, fargs: List = []):
        self.state['w'] = parameters_to_vector(parameters)

        gw = -self._grad_fn(self.state['w'], *fargs)
        fw = self._loss_fn(self.state['w'], *fargs)
        p = gw  # conjugate direction
        p2 = p.T @ p
        self.state.update({
            'lamb': 1e-6,
            'lamb_h': 0,
            'success': True,
            'n_successes': 0,
            'n_vars': len(self.state['w']),
            'gw': gw,
            'fw': fw,
            'p': p,
            'p2': p2,
            'delta': 0,
            'reason': 'not yet converged',
        })
        self._initialized = True
        self.i = 0

    def update(self, model: nn.Module, fargs: List = []):
        if not self._initialized:
            self.init(model.trainable_parameters(), fargs)

        self.compute(fargs)
        self.state['step'] = self.step + 1
        model.update(vector_to_parameters(
            self.state['w'], model.trainable_parameters()))

    def compute(self, fargs: List = []):

        sigma0 = self.defaults['sigma0']
        lamb_min = self.defaults['lamb_min']
        lamb_max = self.defaults['lamb_max']

        # 1) init
        lamb = self.state['lamb']
        lamb_h = self.state['lamb_h']
        success = self.state['success']
        n_successes = self.state['n_successes']
        n_vars = self.state['n_vars']
        gw = self.state['gw']
        fw = self.state['fw']
        p = self.state['p']
        p2 = self.state['p2']
        delta = self.state['delta']

        w = self.state['w']

        # 2) calculate second order info
        if success:
            sigma = sigma0 / mx.sqrt(p2)
            g_small_step = -self._grad_fn(w + sigma * p, *fargs)

            # Hessian matrix times a vector
            delta = p.T @ (gw - g_small_step) / sigma

            if p2 < sys.float_info.epsilon:
                self.state['reason'] = 'limit on machine precision'
                return True

        # 3) scale delta
        delta = delta + (lamb - lamb_h) * p2

        # 4) make Hessian positive definite
        if delta <= 0:
            lamb_h = 2 * (lamb - delta / p2)
            delta = - delta + lamb * p2
            lamb = lamb_h

        # 5) calculate step size
        mu = p.T @ gw
        alpha = mu / delta
        walpha = w + alpha * p
        fwalpha = self._loss_fn(walpha, *fargs)

        # 6) calculate comparison parameter
        Delta = 2 * delta * (fw - fwalpha) / mu**2

        if not mx.isnan(Delta) and Delta >= 0:
            success = True
            lamb_h = 0
            n_successes += 1

            fw = fwalpha
            g_prev = copy.deepcopy(gw)
            w[:] = walpha
            gw[:] = -self._grad_fn(walpha, *fargs)
            if mx.isnan(gw).any() or mx.all(gw == 0):
                self.state['reason'] = 'zero gradient'
                return True

            # restart algorithm every len(w) iterations
            if n_successes % n_vars == 0:
                p = gw
                n_successes = 0
            else:  # update search direction using Polak-Ribière formula
                gamma = (gw @ gw - gw @ g_prev) / (g_prev @ g_prev)
                p = gw + gamma * p
            p2 = p.T @ p

            # reduce scale parameter
            if Delta >= 0.75:
                lamb = max(0.25 * lamb, lamb_min)
        else:
            success = False
            lamb_h = lamb

        # 8) increase scale parameter
        if Delta < 0.25 and p2 != 0:
            lamb = min(lamb + delta * (1 - Delta) / p2, lamb_max)

        self.state.update(
            {
                'w': w,
                'lamb':  lamb,
                'lamb_h': lamb_h,
                'success': success,
                'n_successes': n_successes,
                'gw': gw,
                'fw': fw,
                'p': p,
                'p2': p2,
                'delta': delta
            }
        )
        return False


def simple():
    print('----SIMPLE----')

    def loss_fn(w):
        return (w - 1.5)**2

    def grad_fn(w):
        return 2 * (w - 1.5)

    parameters = {'linear': mx.array([-5.5])}

    scg = SCG(loss_fn, grad_fn)
    scg.init(parameters, fargs=[])

    for i in range(1000):
        scg.state['step'] = scg.step + 1
        done = scg.compute(fargs=[])
        if done:
            break
    print(scg.state['w'], scg.state['reason'], scg.step)


def rosenbrock():
    import matplotlib.pyplot as plt
    from mlx.optimizers import Adam, SGD
    from tqdm import tqdm
    print('----ROSENBROCK----')

    def rosenbrock(xy):
        x, y = xy
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def grad_fn(xy):
        return mx.grad(rosenbrock)(xy)

    def run_optimization(xy_init, optimizer_class, n_iter, **optimizer_kwargs):
        xy_t = mx.array(xy_init, dtype=mx.float32)
        optimizer = optimizer_class(**optimizer_kwargs)
        if isinstance(optimizer, SCG):
            optimizer.init({'o': xy_t})

        path = [xy_t]

        for _ in tqdm(range(n_iter)):
            if isinstance(optimizer, SCG):
                optimizer.state['step'] = optimizer.step + 1
                optimizer.compute()
                xy_t = copy.deepcopy(optimizer.state['w'])
            else:
                loss = rosenbrock(xy_t)
                grad = grad_fn(xy_t)
                xy_t = optimizer.apply_gradients({'o': grad}, {'o': xy_t})['o']
            path.append(xy_t)

        return mx.array(path)

    def plot_rosenbrok(paths, names, colors):
        assert len(paths) == len(names) == len(colors), ValueError
        n = 300
        x = mx.linspace(-2.5, 1.5, n)
        y = mx.linspace(-1.5, 3.5, n)
        minimum = (1.0, 1.0)

        X, Y = mx.meshgrid(x, y)
        Z = rosenbrock([X, Y])

        fig = plt.figure(figsize=(8, 5))

        ax = fig.add_subplot(1, 1, 1)
        ax.contour(X, Y, Z, levels=40, cmap='inferno')
        ax.contourf(X, Y, Z, levels=40, cmap='binary', alpha=0.7)

        for path, name, color in zip(paths, names, colors):
            iter_x, iter_y = path[:, 0], path[:, 1]
            ax.plot(iter_x, iter_y, marker='x', ms=3,
                    lw=2, label=name, color=color)
        ax.legend(fontsize=12)
        ax.axis('off')
        ax.plot(*minimum, 'kD')
        ax.set_title(
            'Rosenbrok Function: $f(x, y) = (1 - x)^2 + 100(y - x^2)^2$')

        fig.tight_layout()
        # fig.savefig(f'media/rosenbrock.png', dpi=300, bbox_inches='tight')
        plt.show()

    xy_init = (-2, 2)
    n_iter = 750

    path_adam = run_optimization(xy_init, Adam, n_iter, learning_rate=0.05)
    path_sgd = run_optimization(xy_init, SGD, n_iter, learning_rate=0.001)
    path_scg = run_optimization(xy_init, SCG, n_iter,
                                loss_fn=rosenbrock, grad_fn=grad_fn)

    freq = 1

    paths = [path_adam[::freq], path_sgd[::freq], path_scg[::freq]]
    names = ['Adam', 'SGD', 'SCG']
    colors = ['royalblue', 'seagreen', 'red']
    print('saving results to rosenbrock.png')
    plot_rosenbrok(paths, names, colors)

    print('[SCG]\n', path_scg[-3:])
    print('[SGD]\n', path_sgd[-3:])
    print('[Adam]\n', path_adam[-3:])


def mnist():
    print('----MNIST----')
    import mlx.nn as nn
    import matplotlib.pyplot as plt

    from mlx.optimizers import Adam, Lion
    from tqdm import tqdm
    # custom modules
    from models import Network
    from dataset import mnist

    mx.random.seed(37)

    train_data, test_data = mnist(-1)

    full_batch_train = next(train_data)
    Xtrain = mx.array(full_batch_train['image'])
    Ttrain = mx.array(full_batch_train['label'])

    full_batch_test = next(test_data)
    Xtest = mx.array(full_batch_test['image'])
    Ttest = mx.array(full_batch_test['label'])

    kwargs = {'n_inputs': Xtrain.shape[1:],
              #   'conv_layers_list': [{'filters': 4, 'kernel_size': 3},
              #                        {'filters': 8, 'kernel_size': 3}],
              'n_hiddens_list': [0],
              'n_outputs': 10,
              'activation_f': 'tanh'
              }

    def accuracy(Y, T):
        return mx.mean(mx.argmax(Y, axis=1) == T)

    EPOCHS = 2000

    def train_and_evaluate(optimizer, model):
        state = [model.state, optimizer.state]

        losses = []
        for _ in tqdm(range(EPOCHS)):
            if isinstance(optimizer, SCG):
                optimizer.update(model, fargs=[Xtrain, Ttrain])
                losses.append(optimizer.state['fw'].item())
            else:
                loss, grads = loss_and_grad_fn(Xtrain, Ttrain)
                losses.append(loss.item())
                optimizer.update(model, grads)
            mx.eval(state)

        train_accuracy = (accuracy(model(Xtrain), Ttrain) * 100).item()
        test_accuracy = (accuracy(model(Xtest), Ttest) * 100).item()
        print(f'train: {train_accuracy:.2f}%')
        print(f'test: {test_accuracy:.2f}%')

        return losses, test_accuracy

    def loss_fn(X, T):
        return nn.losses.cross_entropy(model(X), T, reduction='mean')

    print('[SCG] training...')
    model = Network(**kwargs)
    model.summary()
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def temp_loss_fn(w, X, T):
        old_w = copy.deepcopy(model.trainable_parameters())
        model.update(vector_to_parameters(w, model.trainable_parameters()))
        loss = loss_fn(X, T)
        model.update(old_w)
        return loss

    def temp_grad_fn(w, X, T):
        old_w = copy.deepcopy(model.trainable_parameters())
        model.update(vector_to_parameters(w, model.trainable_parameters()))
        grads = parameters_to_vector(loss_and_grad_fn(X, T)[1])
        model.update(old_w)
        return grads

    scg = SCG(temp_loss_fn, temp_grad_fn)
    scg_loss, scg_acc = train_and_evaluate(scg, model)

    # print('[SGD] training...')
    # model = Network(**kwargs)
    # loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    # sgd = SGD(learning_rate=0.05, momentum=0.9, nesterov=False)
    # sgd_loss, sgd_acc = train_and_evaluate(sgd, model)

    print('[Lion] training...')
    model = Network(**kwargs)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    lion = Lion(learning_rate=0.001)
    lion_loss, lion_acc = train_and_evaluate(lion, model)

    print('[Adam] training...')
    model = Network(**kwargs)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    adam = Adam(learning_rate=0.001)
    adam_loss, adam_acc = train_and_evaluate(adam, model)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for d, a, c, l in zip([adam_loss, lion_loss, scg_loss],
                          [adam_acc, lion_acc, scg_acc],
                          ['royalblue', 'seagreen', 'red'],
                          ['Adam', 'Lion', 'SCG']):
        ax.plot(mx.exp(-mx.array(d)), color=c, lw=2, label=f'{l} ({a:.2f}%)')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Likelihood', fontsize=12)
    ax.set_title(
        f'[MNIST] Full Batch -- {kwargs['n_hiddens_list']}',
        fontsize=12)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.savefig(f'media/mnist.png', dpi=300, bbox_inches='tight')

    plt.show(block=True)


if __name__ == '__main__':
    simple()
    rosenbrock()
    mnist()
