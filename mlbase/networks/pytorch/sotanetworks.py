# Jul 18, 2022
import numpy as np
import torch
import time
import copy

import torch.nn.functional as F
import torch_optimizer as optim


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

    def __init__(self, model_name, n_inputs, n_outputs, feature_extract=False,
                 weights=None, use_gpu=True, seed=None):
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
        self.model_name = model_name.lower()
        self.n_outputs = n_outputs
        self._initialize_model(
            self.model_name, self.n_outputs, feature_extract, weights)
        self.upsample = torch.nn.UpsamplingBilinear2d(
            size=self.input_size) if n_inputs != self.input_size else None

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
        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        self.train_error_trace = []
        self.val_error_trace = []
        self.n_epochs = None
        self.batch_size = None
        self.ridge_penalty = 0
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.model_name}, {self.n_outputs},'
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

    def _set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def _initialize_model(self, model_name, n_outputs, feature_extract, weights):
        """
        :param feature_extract: False, the model is finetuned and all model parameters are updated.
                                True, only the last layer parameters are updated, the others remain fixed.
        """
        try:
            self.model = torch.hub.load(
                'pytorch/vision', model_name, weights=weights)
        except:
            raise Exception('Invalid model name, exiting...')

        self.input_size = 224

        if 'resnet' in model_name:
            """ Resnet50/18/etc.
            """
            self._set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, n_outputs)

        elif 'squeezenet' in model_name:
            """ Squeezenet
            """
            self._set_parameter_requires_grad(self.model, feature_extract)
            self.model.classifier[1] = torch.nn.Conv2d(
                512, n_outputs, kernel_size=(1, 1), stride=(1, 1))
            self.model.num_classes = n_outputs

        elif 'densenet' in model_name:
            """ Densenet
            """
            self._set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(num_ftrs, n_outputs)

        elif 'inception' in model_name:
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self._set_parameter_requires_grad(self.model, feature_extract)
            # Handle the auxilary net
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = torch.nn.Linear(num_ftrs, n_outputs)
            self.model.aux_logits = False
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, n_outputs)
            self.input_size = 299

        elif any(map(model_name.__contains__, ['efficientnet', 'convnext', 'mobilenet',
                                               'alexnet', 'vgg'])):
            """ 
            EfficientNet b0/b1/etc.
            ConvNeXt tiny/small/base/large.
            MobileNet v2/v3_small/etc.
            AlexNet
            vgg16
            """
            self._set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_outputs)

        else:
            raise Exception('Invalid model name, exiting...')

    def summary(self):
        print(self.model)
        print(f'Trainable Params: {self.trainable_params}')

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

    def _upsample_tensor(self, X):
        if self.upsample is not None:
            return self.upsample(X)

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

        running_loss = 0
        for X, T in self._make_batches(Xtrain, Ttrain):
            # overlapping transfer if pinned memory
            X = X.to(self.device, non_blocking=True)
            T = T.to(self.device, non_blocking=True)
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

        Xtrain = self._upsample_tensor(Xtrain)

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
            Xval = self._standardizeX(self._upsample_tensor(Xval))

        if self.loss is not None and self.classification:
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
                # optim.Adahessian,
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
            ]
            names = [str(o.__name__).lower() for o in optimizers]
            try:
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
        self.training_time = time.time() - start_time

        # remove data from gpu, needed?
        Xtrain, Ttrain = list(
            map(lambda x: x.detach().cpu().numpy(), [Xtrain, Ttrain]))
        if validation_data is not None:
            Xval, Tval = list(
                map(lambda x: x.detach().cpu().numpy(), [Xval, Tval]))
        torch.cuda.empty_cache()

    def use(self, X):
        # turn off gradients and other aspects of training
        self.model.eval()
        try:
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X = torch.from_numpy(X).float()

                X = self._standardizeX(self._upsample_tensor(X))
                Ys = []
                for x in self._make_batches(X):
                    x = x.to(self.device)
                    Y = self.model(x)
                    Ys.append(self._unstandardizeT(Y.detach().cpu()).numpy())
                Ys = np.vstack(Ys)
        except RuntimeError:
            raise
        finally:
            torch.cuda.empty_cache()
        return Ys


class NeuralNetworkClassifier(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        # TODO: only supporting CrossEntropyLoss as use function now computes softmax
        # self.loss = torch.nn.NLLLoss()
        # CrossEntropyLoss computes LogSoftmax then NLLLoss
        self.classification = True
        self.loss = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)

    def use(self, X, all_output=False):
        """
        Return:
            if all_output: predicted classes, all layers + softmax
            else: predicted classes
        """
        # TODO: add batching to inference
        # turn off gradients and other aspects of training
        self.model.eval()
        try:
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X = torch.from_numpy(X).float()
                X = self._standardizeX(self._upsample_tensor(X))
                Ys = []
                for x in self._make_batches(X):
                    x = x.to(self.device)
                    Y = self.model(x)
                    Ys.append(F.softmax(Y, dim=1).detach().cpu().numpy())
                Ys = np.vstack(Ys)
        except RuntimeError:
            raise
        finally:
            torch.cuda.empty_cache()
        Y = Ys.argmax(1).reshape(-1, 1)
        return (Y, Ys) if all_output else Y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    def rmse(A, B): return np.sqrt(np.mean((A - B)**2))
    def accuracy(A, B): return 100. * np.mean(A == B)
    br = ''.join(['-']*8)

    print(f'{br}Testing NeuralNetwork for CNN regression{br}')
    #---------------------------------------------------------------#
    X = np.zeros((100, 3, 10, 10))
    T = np.zeros((100, 1), dtype=int)
    for i in range(X.shape[0]):
        col = i // 10
        X[i, :, 0: col + 1, 0] = 1
        T[i, 0] = col + 1

    nnet = NeuralNetwork('alexnet', X.shape[1:], T.shape[1],
                         feature_extract=False, weights=None,
                         use_gpu=True, seed=1234)
    nnet.summary()
    nnet.train(X, T, n_epochs=50, batch_size=32, learning_rate=0.0001,
               opt='adam', verbose=True, shuffle=True)

    Y = nnet.use(X)
    print(f'rmse={rmse(T, Y):.3f}')
    plt.figure(1)
    plt.plot(nnet.train_error_trace)

    print(f'{br}Testing NeuralNetwork for CNN classification{br}')
    #---------------------------------------------------------------#
    X = np.zeros((100, 3, 10, 10))
    T = np.zeros((100, 1))
    for i in range(100):
        col = i // 10
        X[i, 0, :, 0: col + 1] = 1
        # TODO: class must be between [0, num_classes-1]
        T[i, 0] = 0 if col < 3 else 1 if col < 7 else 2

    nnet = NeuralNetworkClassifier('alexnet', X.shape[1:], len(np.unique(T)),
                                   feature_extract=False, weights=None,
                                   use_gpu=True, seed=1234)
    nnet.summary()
    nnet.train(X, T, n_epochs=50, batch_size=32, learning_rate=0.0001,
               opt='adam', verbose=True, shuffle=True)

    Y = nnet.use(X)
    print(f'Accuracy: {accuracy(Y, T):.3f}')
    plt.figure(2)
    plt.plot(np.exp(-np.array(nnet.train_error_trace)))

    plt.show(block=True)
