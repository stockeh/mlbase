# June 7, 2022
import numpy as np
import torch
import time
import copy

import torch.nn.functional as F
import torch_optimizer as optim

from mlbase.networks.pytorch.optimizers.scg_optim_pytorch import SCG
from mlbase.networks.pytorch.optimizers.alopex_optim_pytorch import Alopex


class EarlyStopping(object):
    '''
    MIT License, Copyright (c) 2018 Stefano Nardo https://gist.github.com/stefanonardo
    es = EarlyStopping(patience=5)

    for epoch in range(n_epochs):
        # train the model for one epoch, on training set
        train_one_epoch(model, data_loader)
        # evalution on dev set (i.e., holdout from training)
        metric = eval(model, data_loader_dev)
        if es.step(metric):
            break  # early stop criterion is met, we can stop now
    '''

    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


class NeuralNetwork():

    class Network(torch.nn.Module):
        def __init__(self, n_inputs, n_hiddens_list, n_outputs, conv_layers, activation_f):
            super().__init__()
            if not isinstance(n_hiddens_list, list):
                raise Exception(
                    'Network: n_hiddens_list must be a list.')

            if len(n_hiddens_list) == 0 or n_hiddens_list[0] == 0:
                self.n_hidden_layers = 0
            else:
                self.n_hidden_layers = len(n_hiddens_list)

            # network varaibles
            self.n_inputs = n_inputs
            self.n_hiddens_list = n_hiddens_list
            self.n_outputs = n_outputs
            self.conv_layers = conv_layers

            # build nnet
            self.model = torch.nn.ModuleList()

            activations = [
                torch.nn.Tanh,
                torch.nn.Sigmoid,
                torch.nn.ReLU,
                torch.nn.ELU,
                torch.nn.PReLU,
                torch.nn.ReLU6,
                torch.nn.LeakyReLU,
                torch.nn.Mish,
            ]
            names = [str(o.__name__).lower() for o in activations]
            try:
                activation = activations[names.index(
                    str(activation_f).lower())]
            except:
                raise NotImplementedError(
                    f'__init__: {activation_f=} is not yet implemented.')
            l = 0
            ni = np.asarray(n_inputs)
            # add convolutional layers
            # TODO: currently only works with padding='same' and stride=1
            if self.conv_layers:
                for conv_layer in self.conv_layers:
                    # check if 1d or 2d conv
                    if 'convd' in conv_layer:
                        convd = conv_layer['convd']
                    else:
                        convd = 2  # assume 2D convoltuion
                    n_channels = ni[0]  # C,H,W  or  C,W
                    if convd == 2:  # 2D Conv
                        self.model.add_module(f'conv_{l}', torch.nn.Conv2d(
                            n_channels, conv_layer['n_units'], conv_layer['shape'],
                            stride=1, padding='same', padding_mode='zeros'))
                        self.model.add_module(f'activation_{l}', activation())
                        self.model.add_module(
                            f'maxpool_{l}', torch.nn.MaxPool2d(2, stride=2))
                    elif convd == 1:  # 1D Conv
                        self.model.add_module(f'conv_{l}', torch.nn.Conv1d(
                            n_channels, conv_layer['n_units'], conv_layer['shape'],
                            stride=1, padding='same', padding_mode='zeros'))
                        self.model.add_module(f'activation_{l}', activation())
                        self.model.add_module(
                            f'maxpool_{l}', torch.nn.MaxPool1d(2, stride=2))
                    # TODO: currently only to divide H, W dimensions by 2
                    # with 'same' padding
                    ni = np.concatenate([[conv_layer['n_units']], ni[1:] // 2])
                    l += 1

            if ni.ndim > 0:  # only the case with vectorized input features
                self.model.add_module('flatten', torch.nn.Flatten())  # okay
            ni = np.prod(ni)

            # add fully-connected layers
            if self.n_hidden_layers > 0:
                for i, n_units in enumerate(n_hiddens_list):
                    self.model.add_module(
                        f'linear_{l}', torch.nn.Linear(ni, n_units))
                    self.model.add_module(f'activation_{l}', activation())
                    # if self.conv_layers:
                    #     self.model.add_module(
                    #         f'dropout_{l}', torch.nn.Dropout(0.2))
                    ni = n_units
                    l += 1
            self.model.add_module(
                f'output', torch.nn.Linear(ni, n_outputs))

            # self.model.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        def forward_all_outputs(self, X):
            Ys = [X]
            for i, layer in enumerate(self.model):
                Ys.append(layer(Ys[i]))
            return Ys[1:]  # all outputs without original inputs

        def forward(self, X):
            Ys = self.forward_all_outputs(X)
            return Ys[-1]

    def __init__(self, n_inputs, n_hiddens_list, n_outputs, conv_layers=[],
                 activation_f='tanh', use_gpu=True, seed=None):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
        self.seed = seed

        if use_gpu and not torch.cuda.is_available():
            print('\nGPU is not available. Running on CPU.\n')
            use_gpu = False
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.classification = False

        # Build nnet
        self.model = self.Network(
            n_inputs, n_hiddens_list, n_outputs, conv_layers, activation_f)
        self.model.to(self.device)
        self.loss = None
        self.optimizer = None

        # Member variables for standardization
        self.standardize_x = True
        self.standardize_t = True
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        # Bookkeeping
        self.train_error_trace = []
        self.val_error_trace = []
        self.n_epochs = None
        self.batch_size = None
        self.ridge_penalty = 0
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.model.n_inputs}, {self.model.n_hiddens_list}, {self.model.n_outputs},'
        str += f' {self.use_gpu=}, {self.seed=})'
        if self.training_time is not None:
            str += f'\n   Network was trained for {self.n_epochs} epochs'
            str += f' that took {self.training_time:.4f} seconds.\n   Final objective values...'
            str += f' train: {self.train_error_trace[-1]:.3f},'
            if len(self.val_error_trace):
                str += f'val: {self.val_error_trace[-1]:.3f}'
        else:
            str += '  Network is not trained.'
        return str

    def summary(self):
        print(self.model)
        print(
            f'Trainable Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    def _standardizeX(self, X):
        if self.standardize_x:
            result = (X - self.Xmeans) / self.XstdsFixed
            result[:, self.Xconstant] = 0.0
            return result
        else:
            return X

    def _unstandardizeX(self, Xs):
        return (self.Xstds * Xs + self.Xmeans) if self.standardize_x else Xs

    def _standardizeT(self, T):
        if self.standardize_t:
            result = (T - self.Tmeans) / self.TstdsFixed
            result[:, self.Tconstant] = 0.0
            return result
        else:
            return T

    def _unstandardizeT(self, Ts):
        return (self.Tstds * Ts + self.Tmeans) if self.standardize_t else Ts

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

    def _get_standardize_stats(self):
        o = None
        if self.Xmeans is not None:
            o = (self.Xmeans, self.Xstds, self.Xconstant, self.XstdsFixed,
                 self.Tmeans, self.Tstds, self.Tconstant, self.TstdsFixed)
        return o

    def _make_batches(self, X, T=None):
        if self.batch_size == -1:
            if T is None:
                yield X
            else:
                yield X, T
        else:
            for i in range(0, X.shape[0], self.batch_size):
                if T is None:
                    yield X[i:i+self.batch_size]
                else:
                    yield X[i:i+self.batch_size], T[i:i+self.batch_size]

    def _train(self, training_data, validation_data):
        # training
        #---------------------------------------------------------------#
        Xtrain, Ttrain = training_data
        self.model.train()

        def _l2_regularization(T):
            # https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L10_regularization_slides.pdf
            # https://towardsdatascience.com/understanding-the-scaling-of-l²-regularization-in-the-context-of-neural-networks-e3d25f8b50db
            if self.ridge_penalty <= 0:
                return 0
            penalty = 0
            for name, p in self.model.named_parameters():
                if 'weight' in name:
                    penalty += p.pow(2.0).sum()
            # lambda / n * sum(w**2)
            return self.ridge_penalty / T.shape[0] * penalty

        if isinstance(self.optimizer, SCG):  # only for SGD

            def closure():
                running_loss = 0
                self.optimizer.zero_grad()
                for X, T in self._make_batches(Xtrain, Ttrain):
                    # overlapping transfer if pinned memory
                    X = X.to(self.device, non_blocking=True)
                    T = T.to(self.device, non_blocking=True)
                    Y = self.model(X)
                    loss = (self.loss(Y, T) + _l2_regularization(T)) / \
                        self.n_train_batches
                    loss.backward()
                    running_loss += loss.item()
                return running_loss

            loss = self.optimizer.step(closure)
            self.train_error_trace.append(loss)

            self.scg_state.append(
                {i: self.optimizer.state[i] for i in self.optimizer.state if i != 'losses'})

        else:
            running_loss = 0
            for X, T in self._make_batches(Xtrain, Ttrain):
                # overlapping transfer if pinned memory
                X = X.to(self.device, non_blocking=True)
                T = T.to(self.device, non_blocking=True)

                if isinstance(self.optimizer, Alopex):
                    def closure():
                        Y = self.model(X)
                        return self.loss(Y, T) + _l2_regularization(T)
                    loss = self.optimizer.step(closure)
                else:
                    # compute prediction error
                    Y = self.model(X)
                    loss = self.loss(Y, T) + _l2_regularization(T)
                    # backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # unaveraged sum of losses over all samples
                # https://discuss.pytorch.org/t/interpreting-loss-value/17665/10
                running_loss += loss.item() / self.n_train_batches

            # maintain loss over every epoch
            self.train_error_trace.append(running_loss)

        # validation
        #---------------------------------------------------------------#
        if validation_data is not None:
            Xval, Tval = validation_data
            running_loss = 0
            self.model.eval()
            with torch.no_grad():
                for X, T in self._make_batches(Xval, Tval):
                    # overlapping transfer if pinned memory
                    X = X.to(self.device, non_blocking=True)
                    T = T.to(self.device, non_blocking=True)
                    Y = self.model(X)
                    loss = self.loss(Y, T) + _l2_regularization(T)
                    running_loss += loss.item() / self.n_val_batches

                self.val_error_trace.append(running_loss)

    def train(self, Xtrain, Ttrain, n_epochs, batch_size, learning_rate,
              opt='adam', weight_decay=0, ridge_penalty=0, early_stopping=False,
              validation_data=None, shuffle=False, verbose=True, standardize_x=True,
              standardize_t=True):

        if not isinstance(Xtrain, torch.Tensor):
            Xtrain, Ttrain = list(map(lambda x: torch.from_numpy(
                x).float(), [Xtrain, Ttrain]))

        self.standardize_x = standardize_x
        self.standardize_t = standardize_t if not self.classification else False
        self._setup_standardize(Xtrain, Ttrain)  # only occurs once
        Xtrain = self._standardizeX(Xtrain)

        if validation_data is not None:
            assert len(
                validation_data) == 2, 'validation_data: must be (Xval, Tval).'
            Xval, Tval = validation_data[0], validation_data[1]
            if verbose and not self.classification and standardize_t:
                print(f'{Tval.mean()=:.3f}, {Tval.std()=:.3f}')
            if not isinstance(Xval, torch.Tensor):
                Xval, Tval = list(map(lambda x: torch.from_numpy(
                    x).float(), [Xval, Tval]))
            Xval = self._standardizeX(Xval)

        if self.classification:
            if isinstance(self.loss, torch.nn.CrossEntropyLoss):
                Ttrain = Ttrain.flatten().type(torch.LongTensor)
                if validation_data is not None:
                    Tval = Tval.flatten().type(torch.LongTensor)
        else:  # standardize targets
            Ttrain = self._standardizeT(Ttrain)
            if validation_data is not None:
                Tval = self._standardizeT(Tval)

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ridge_penalty = ridge_penalty

        if self.loss is None:
            self.loss = torch.nn.MSELoss()
        self.loss = self.loss.to(self.device)

        if self.optimizer is None:
            optimizers = [
                torch.optim.SGD,
                torch.optim.Adam,
                torch.optim.RMSprop,
                optim.A2GradExp,
                optim.A2GradInc,
                optim.A2GradUni,
                optim.AccSGD,
                optim.AdaBelief,
                optim.AdaBound,
                optim.AdaMod,
                optim.Adafactor,
                optim.AdamP,
                optim.AggMo,
                optim.Apollo,
                optim.DiffGrad,
                optim.Lamb,
                optim.NovoGrad,
                optim.PID,
                optim.QHAdam,
                optim.QHM,
                optim.RAdam,
                optim.Ranger,
                optim.RangerQH,
                optim.RangerVA,
                optim.SGDP,
                optim.SGDW,
                optim.SWATS,
                optim.Yogi,
                SCG,
                Alopex,
            ]
            names = [str(o.__name__).lower() for o in optimizers]
            try:
                if str(opt).lower() == 'scg':  # no learning rate
                    self.optimizer = optimizers[names.index(str(opt).lower())](
                        self.model.parameters())
                    self.scg_state = []
                elif str(opt).lower() == 'alopex':
                    # TODO: what to do with `temp_iter`...
                    N = 1 if batch_size == - \
                        1 else Xtrain.shape[0] // batch_size
                    self.optimizer = optimizers[names.index(str(opt).lower())](
                        self.model.parameters(), eta0=learning_rate, N=N)
                else:
                    self.optimizer = optimizers[names.index(str(opt).lower())](
                        self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            except:
                raise NotImplementedError(
                    f'train: {opt=} is not yet implemented.')

        # compute mini-batch sizes
        _minibatch = batch_size != - \
            1 and batch_size != Xtrain.shape[0]
        # train
        train_bs = batch_size if _minibatch else Xtrain.shape[0]
        self.n_train_batches = (
            Xtrain.shape[0] + train_bs - 1) // train_bs
        # val
        if validation_data is not None:
            val_bs = batch_size if _minibatch else Xval.shape[0]
            self.n_val_batches = (
                Xval.shape[0] + val_bs - 1) // val_bs

        print_every = n_epochs // 10 if n_epochs > 9 else 1
        if early_stopping:
            es = EarlyStopping(patience=5)
        # training loop
        #---------------------------------------------------------------#
        start_time = time.time()
        for epoch in range(n_epochs):
            if shuffle:  # shuffle after every epoch
                if self.seed is not None:
                    torch.manual_seed(self.seed + epoch)
                train_inds = torch.randperm(Xtrain.size()[0])
                Xtrain = Xtrain[train_inds]
                Ttrain = Ttrain[train_inds]
            # forward, grad, backprop
            self._train((Xtrain, Ttrain), (Xval, Tval)
                        if validation_data is not None else None)
            if early_stopping and validation_data is not None and es.step(self.val_error_trace[-1]):
                self.n_epochs = epoch + 1
                break  # early stop criterion is met, we can stop now
            if verbose and (epoch + 1) % print_every == 0:
                st = f'Epoch {epoch + 1} error - train: {self.train_error_trace[-1]:.5f},'
                if validation_data is not None:
                    st += f' val: {self.val_error_trace[-1]:.5f}'
                print(st)
            # print(self.optimizer.state['n_successes'],
            #       '/', self.optimizer.state['n_vars'])
        self.training_time = time.time() - start_time

        # remove data from gpu, needed?
        Xtrain, Ttrain = list(
            map(lambda x: x.detach().cpu().numpy(), [Xtrain, Ttrain]))
        if validation_data is not None:
            Xval, Tval = list(
                map(lambda x: x.detach().cpu().numpy(), [Xval, Tval]))
        torch.cuda.empty_cache()

        # convert loss to likelihood
        # TODO: append values to continue with training
        # if self.classification:
        #     self.train_error_trace = np.exp(
        #         -np.asarray(self.train_error_trace))
        #     if validation_data is not None:
        #         self.val_error_trace = np.exp(
        #             -np.asarray(self.val_error_trace))

        # return self  # this was causing memory leakes

    def use(self, X, all_output=False, detach=True):
        # turn off gradients and other aspects of training
        self.model.eval()
        try:
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X = torch.from_numpy(X).float()
                X = self._standardizeX(X)
                Ys = None
                i = 0
                nsamples = X.shape[0]
                for x in self._make_batches(X):
                    x = x.to(self.device)
                    end = x.shape[0] if x.shape[0] < nsamples else nsamples
                    Y = self.model.forward_all_outputs(x)
                    if detach:
                        Y = [y.detach().cpu().numpy() for y in Y]
                    Y[-1] = self._unstandardizeT(Y[-1])
                    if Ys is None:
                        Ys = [np.zeros((nsamples, *y.shape[1:])) for y in Y]
                    for j in range(len(Ys)):
                        Ys[j][i:i+end] = Y[j]
                    i += end
        except RuntimeError:
            raise
        finally:
            torch.cuda.empty_cache()
        return Ys if all_output else Ys[-1]


class NeuralNetworkClassifier(NeuralNetwork):

    class Network(NeuralNetwork.Network):
        def __init__(self, *args, **kwargs):
            super(self.__class__, self).__init__(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.classification = True
        if isinstance(self.model.n_outputs, int) and self.model.n_outputs == 2:
            self.model.n_outputs = 1
            ftrs = self.model.model.output.in_features
            self.model.model.output = torch.nn.Linear(ftrs, 1, bias=True)
            self.loss = torch.nn.BCEWithLogitsLoss()  # computes Sigmoid then BCELoss
            self.model.to(self.device)
        else:
            self.loss = torch.nn.CrossEntropyLoss()  # computes LogSoftmax then NLLLoss

    def use(self, X, all_output=False, probs=False):
        """
        Return:
            if all_output: predicted classes, all layers + softmax
            else: predicted classes
        """
        # turn off gradients and other aspects of training
        def probf(l): return torch.sigmoid(
            l) if l.shape[1] == 1 else F.softmax(l, dim=1)

        def maxf(p): return np.where(
            p > 0.5, 1, 0) if p.shape[1] == 1 else p.argmax(1)

        self.model.eval()
        try:
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X = torch.from_numpy(X).float()
                X = self._standardizeX(X)
                if all_output:
                    i = 0
                    Ys = None
                    nsamples = X.shape[0]
                p = []
                for x in self._make_batches(X):
                    x = x.to(self.device)
                    if all_output:
                        end = x.shape[0] if x.shape[0] < nsamples else nsamples
                        Y = self.model.forward_all_outputs(x)
                        logits = Y[-1]
                        Y = [y.detach().cpu().numpy() for y in Y]
                        if Ys is None:
                            Ys = [np.zeros((nsamples, *y.shape[1:]))
                                  for y in Y]
                        for j in range(len(Ys)):
                            Ys[j][i:i+end] = Y[j]
                        i += end
                    else:
                        logits = self.model(x)
                    p.append(probf(logits).detach().cpu().numpy())
                p = np.vstack(p)
        except RuntimeError:
            raise
        finally:
            torch.cuda.empty_cache()
        Y = maxf(p).reshape(-1, 1)

        if all_output:
            return (Y, Ys + [p])
        elif probs:
            return (Y, p)
        else:
            return Y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    def rmse(A, B): return np.sqrt(np.mean((A - B)**2))
    def accuracy(A, B): return 100. * np.mean(A == B)
    br = ''.join(['-']*8)

    print(f'{br}Testing NeuralNetwork for regression{br}')
    #---------------------------------------------------------------#
    X = np.arange(100).reshape((-1, 1))
    T = np.sin(X * 0.04)

    n_hiddens_list = [10, 10]

    nnet = NeuralNetwork(X.shape[1], n_hiddens_list,
                         T.shape[1], activation_f='tanh')
    nnet.summary()
    nnet.train(X, T, n_epochs=1000, batch_size=32,
               learning_rate=0.01, opt='sgd')
    Y = nnet.use(X)

    print(f'RMSE: {rmse(T, Y):.3f}')
    plt.figure(1)
    plt.plot(nnet.train_error_trace)

    print(f'{br}Testing NeuralNetwork for CNN regression{br}')
    #---------------------------------------------------------------#
    # TODO: requires C, H, W dimensions
    X = np.zeros((100, 1, 10, 10))
    T = np.zeros((100, 1))
    for i in range(100):
        col = i // 10
        X[i, :, 0: col + 1, 0] = 1
        T[i, 0] = col + 1

    conv_layers = [{'n_units': 1, 'shape': [3, 3]},
                   {'n_units': 1, 'shape': [3, 3]}]
    n_hiddens_list = [10]

    nnet = NeuralNetwork(X.shape[1:], n_hiddens_list,
                         T.shape[1], conv_layers, activation_f='tanh')
    nnet.summary()
    nnet.train(X, T, n_epochs=1000, batch_size=32,
               learning_rate=0.001, opt='adam')
    Y = nnet.use(X)

    print(f'RMSE: {rmse(T, Y):.3f}')
    plt.figure(2)
    plt.plot(nnet.train_error_trace)

    print(f'{br}Testing NeuralNetwork for CNN classification (BCE){br}')
    #---------------------------------------------------------------#
    X = np.zeros((100, 1, 10, 10))
    T = np.zeros((100, 1))
    for i in range(100):
        col = i // 10
        X[i, 0, :, 0: col + 1] = 1
        # TODO: class must be between [0, num_classes-1]
        T[i, 0] = 0 if col < 5 else 1

    n_hiddens_list = [5]*2
    conv_layers = [{'n_units': 3, 'shape': 3},
                   {'n_units': 1, 'shape': [3, 3]}]

    nnet = NeuralNetworkClassifier(X.shape[1:], n_hiddens_list, len(
        np.unique(T)), conv_layers, use_gpu=True, seed=None)
    nnet.summary()
    print(nnet.loss)
    nnet.train(X, T, validation_data=None,
               n_epochs=50, batch_size=32, learning_rate=0.01, opt='adam',  # accsgd
               ridge_penalty=0, verbose=True)
    Y = nnet.use(X)
    print(f'Accuracy: {accuracy(Y, T):.3f}')
    plt.figure(3)
    plt.plot(nnet.train_error_trace)

    print(f'{br}Testing NeuralNetwork for CNN classification (NLL){br}')
    #---------------------------------------------------------------#
    X = np.zeros((100, 1, 10, 10))
    T = np.zeros((100, 1))
    for i in range(100):
        col = i // 10
        X[i, 0, :, 0: col + 1] = 1
        # TODO: class must be between [0, num_classes-1]
        T[i, 0] = 0 if col < 3 else 1 if col < 7 else 2

    n_hiddens_list = [5]*2
    conv_layers = [{'n_units': 3, 'shape': 3},
                   {'n_units': 1, 'shape': [3, 3]}]

    nnet = NeuralNetworkClassifier(X.shape[1:], n_hiddens_list, len(
        np.unique(T)), conv_layers, use_gpu=True, seed=None)
    nnet.summary()
    print(nnet.loss)
    nnet.train(X, T, validation_data=None,
               n_epochs=50, batch_size=32, learning_rate=0.01, opt='adam',  # accsgd
               ridge_penalty=0, verbose=True)
    Y = nnet.use(X)
    print(f'Accuracy: {accuracy(Y, T):.3f}')
    plt.figure(4)
    plt.plot(nnet.train_error_trace)

    plt.show(block=True)
