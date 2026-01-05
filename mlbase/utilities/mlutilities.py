import matplotlib.collections as pltcoll
import matplotlib.patches as pltpatch  # for Arc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

######################################################################
# Machine Learning Utilities.
#
#  percent_correct
#  rmse
#  r2
#  topk
#  partition
#  confusion_matrix
#  evaluation
#  draw
######################################################################


def percent_correct(Y, T):
    return 100 * np.mean(Y == T)


def rmse(Y, T):
    return np.sqrt(np.mean((Y - T) ** 2))


def r2(T, Y):
    """Order is important"""
    return r2_score(T, Y)


def topk(softmax, T, topks=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topks)
    batch_size = T.shape[0]
    pred = (-softmax).argsort(axis=1)[:, :maxk]  # (N x C)
    correct = pred == T

    res = []
    for k in topks:
        correct_k = correct[:, :k].sum()
        res.append(100 * correct_k / batch_size)
    return res


######################################################################


def regression_summary(
    T, Y, units="", plot=True, xlabel=None, ylabel=None, filename=None
):
    """Return and plot target vs predicted with r2 and rmse.

    !important: order of params matter.

    :param T: numpy array of target values (N,F)
    :param Y: numpy array of predicted values (N,F)
    :return fig: `matplotlib.pyplot` figure
    :return ax: `matplotlib.pyplot` axis
    """
    r2v = r2(T, Y)
    rmsev = rmse(T, Y)
    if plot:
        default_font = 14
        line_width = 2.5

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(T, Y, ".", color="gray")
        mi = min(min(T), min(Y))
        ma = max(max(T), max(Y))

        ax.plot(
            np.arange(mi, ma), np.arange(mi, ma), color="blue", linewidth=line_width
        )

        ax.set_title(
            f"$R^2$: {r2v:.3f}, RMSE: {rmsev:.3f} {units}",
            loc="right",
            fontsize=12,
            fontstyle="italic",
        )
        ax.set_xlabel(
            "Target Value" if xlabel is None else xlabel, fontsize=default_font
        )
        ax.set_ylabel(
            "Predicted Value" if xlabel is None else ylabel, fontsize=default_font
        )
        ax.tick_params(axis="x", labelsize=default_font)
        ax.tick_params(axis="y", labelsize=default_font)

        ax.grid(True)
        fig.tight_layout()

        if filename:
            fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)
            print("---saved")

    return r2v, rmsev


######################################################################


def partition(X, T, trainFraction, shuffle=False, classification=False, seed=1234):
    """Usage: Xtrain,Train,Xvalidate,Tvalidate,Xtest,Ttest = partition(X,T,(0.6,0.2,0.2),shuffle=False,classification=True)
    X is nSamples x nFeatures.
    fractions can have just two values, for partitioning into train and test only
    If classification=True, T is target class as integer. Data partitioned
      according to class proportions.
    """
    # Skip the validation step
    validateFraction = 0
    testFraction = 1 - trainFraction

    rowIndices = np.arange(X.shape[0])
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(rowIndices)

    if not classification:
        # regression, so do not partition according to targets.
        n = X.shape[0]
        nTrain = round(trainFraction * n)
        nValidate = round(validateFraction * n)
        nTest = round(testFraction * n)
        if nTrain + nValidate + nTest > n:
            nTest = n - nTrain - nValidate
        Xtrain = X[rowIndices[:nTrain], :]
        Ttrain = T[rowIndices[:nTrain], :]
        if nValidate > 0:
            Xvalidate = X[rowIndices[nTrain : nTrain + nValidate], :]
            Tvalidate = T[rowIndices[nTrain:nTrain:nValidate], :]
        Xtest = X[rowIndices[nTrain + nValidate : nTrain + nValidate + nTest], :]
        Ttest = T[rowIndices[nTrain + nValidate : nTrain + nValidate + nTest], :]

    else:
        # classifying, so partition data according to target class
        classes = np.unique(T)
        trainIndices = []
        validateIndices = []
        testIndices = []
        for c in classes:
            # row indices for class c
            cRows = np.where(T[rowIndices, :] == c)[0]
            # collect row indices for class c for each partition
            n = len(cRows)
            nTrain = round(trainFraction * n)
            nValidate = round(validateFraction * n)
            nTest = round(testFraction * n)
            if nTrain + nValidate + nTest > n:
                nTest = n - nTrain - nValidate
            trainIndices += rowIndices[cRows[:nTrain]].tolist()
            if nValidate > 0:
                validateIndices += rowIndices[
                    cRows[nTrain : nTrain + nValidate]
                ].tolist()
            testIndices += rowIndices[
                cRows[nTrain + nValidate : nTrain + nValidate + nTest]
            ].tolist()
        Xtrain = X[trainIndices, :]
        Ttrain = T[trainIndices, :]
        if nValidate > 0:
            Xvalidate = X[validateIndices, :]
            Tvalidate = T[validateIndices, :]
        Xtest = X[testIndices, :]
        Ttest = T[testIndices, :]
    if nValidate > 0:
        return Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest
    else:
        return Xtrain, Ttrain, Xtest, Ttest


######################################################################


def confusion_matrix(actual, predicted, classes):
    nc = len(classes)
    confmat = np.zeros((nc, nc))
    for ri in range(nc):
        trues = (actual == classes[ri]).squeeze()
        predictedThisClass = predicted[trues]
        keep = trues
        predictedThisClassAboveThreshold = predictedThisClass
        for ci in range(nc):
            confmat[ri, ci] = np.sum(
                predictedThisClassAboveThreshold == classes[ci]
            ) / float(np.sum(keep))
    print_confusion_matrix(confmat, classes)
    return confmat


def print_confusion_matrix(confmat, classes):
    print("   ", end="")
    for i in classes:
        print("%5d" % (i), end="")
    print("\n    ", end="")
    print("{:s}".format("------" * len(classes)))
    for i, t in enumerate(classes):
        print("{:2d} |".format(t), end="")
        for i1, t1 in enumerate(classes):
            if confmat[i, i1] == 0:
                print("  0  ", end="")
            else:
                print("{:5.1f}".format(100 * confmat[i, i1]), end="")
        print()


######################################################################


def confusion(Y, T, classes=None):
    if classes is None:
        classes = np.unique(T)
    confmat = np.zeros((len(classes), len(classes)), dtype=int)
    for i in range(len(T)):
        confmat[T[i], Y[i]] += 1
    return confmat


def evaluate(Y, T, verbose=True):
    """Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    """
    confmat = confusion(Y, T, np.unique(T))

    with np.errstate(divide="ignore", invalid="ignore"):
        # Success Ratio (SR)
        precision = np.diag(confmat) / np.sum(confmat, axis=0)  # tp / (tp + fp)
        # probability of Detection (POD)
        recall = np.diag(confmat) / np.sum(confmat, axis=1)  # tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)  # per class

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)

    accuracy = np.trace(confmat) / len(T)

    if verbose:
        print_metrics(confmat, precision, recall, f1, accuracy, np.unique(T))

    return confmat, precision, recall, f1, accuracy


def print_confmat(confmat, class_names=None):
    classes = np.arange(confmat.shape[0]) if class_names is None else class_names
    spacing = len(str(np.max(confmat)))
    class_spacing = len(str(np.max(classes))) + 1
    if class_spacing > spacing:
        spacing = class_spacing
    top = " " * (class_spacing) + "".join(
        " {i: < {spacing}}".format(i=i, spacing=str(spacing)) for i in classes
    )
    t = [
        "{c:<{spacing}} |".format(c=classes[j], spacing=str(class_spacing - 1))
        + "".join(" {i:<{spacing}}".format(i=i, spacing=str(spacing)) for i in row)
        for j, row in enumerate(confmat)
    ]
    hdr = " " * class_spacing + "-" * (len(t[0]) - class_spacing)
    print("Confusion Matrix:", top, hdr, "\n".join(t), sep="\n")


def print_metrics(confmat, precision, recall, f1, accuracy, class_names):
    # Print Classes
    wrap = "-" * 40
    print(wrap)
    # print('Classes:', ', '.join(f'{k}: {v}' for k,
    #                             v in self.class_dict.items()), end='\n\n')
    # Print Confusion Matrix
    print_confmat(confmat, class_names=class_names)

    # All-Class Metrics
    labels = ["Precision", "Recall", "F1"]
    precision = np.append(precision, precision.mean())
    recall = np.append(recall, recall.mean())
    f1 = np.append(
        f1, 2 * precision.mean() * recall.mean() / (precision.mean() + recall.mean())
    )
    # Print Metrics
    metrics = np.vstack([precision, recall, f1])
    label_spacing = max([len(l) for l in labels]) + 1
    metric_spacing = max([len(f"{m:.3f}") for m in metrics.flatten()])
    mean = "  mean"
    top = (
        " " * (label_spacing)
        + "".join(
            " {i: < {spacing}}".format(i=i, spacing=str(metric_spacing))
            for i in class_names
        )
        + mean
    )
    t = [
        "{i:<{spacing}}|".format(i=labels[j], spacing=str(label_spacing))
        + "".join(f" {i:.3f}" for i in row)
        for j, row in enumerate(metrics)
    ]
    hdr = " " * label_spacing + "-" * (len(t[0]) - label_spacing)
    print("\nMetrics:", top, hdr, "\n".join(t), sep="\n")
    # Print Accuracy
    print(f"\nOverall Accuracy: {accuracy*100:.3f} %")
    print(wrap)


######################################################################


def draw(Vs, W, inputNames=None, outputNames=None, gray=False):

    def isOdd(x):
        return x % 2 != 0

    W = Vs + [W]
    nLayers = len(W)

    # calculate xlim and ylim for whole network plot
    #  Assume 4 characters fit between each wire
    #  -0.5 is to leave 0.5 spacing before first wire
    xlim = max(map(len, inputNames)) / 4.0 if inputNames else 1
    ylim = 0

    for li in range(nLayers):
        ni, no = W[li].shape  # no means number outputs this layer
        if not isOdd(li):
            ylim += ni + 0.5
        else:
            xlim += ni + 0.5

    ni, no = W[nLayers - 1].shape
    if isOdd(nLayers):
        xlim += no + 0.5
    else:
        ylim += no + 0.5

    # Add space for output names
    if outputNames:
        if isOdd(nLayers):
            ylim += 0.25
        else:
            xlim += round(max(map(len, outputNames)) / 4.0)

    ax = plt.gca()

    character_width_factor = 0.07
    padding = 2
    if inputNames:
        x0 = max([1, max(map(len, inputNames)) * (character_width_factor * 3.5)])
    else:
        x0 = 1
    y0 = 0  # to allow for constant input to first layer
    # First Layer
    if inputNames:
        y = 0.55
        for n in inputNames:
            y += 1
            ax.text(
                x0 - (character_width_factor * padding),
                y,
                n,
                horizontalalignment="right",
                fontsize=20,
            )

    patches = []
    for li in range(nLayers):
        thisW = W[li]
        maxW = np.max(np.abs(thisW))
        ni, no = thisW.shape
        if not isOdd(li):
            # Even layer index. Vertical layer. Origin is upper left.
            # Constant input
            ax.text(x0 - 0.2, y0 + 0.5, "1", fontsize=20)
            for i in range(ni):
                ax.plot((x0, x0 + no - 0.5), (y0 + i + 0.5, y0 + i + 0.5), color="gray")
            # output lines
            for i in range(no):
                ax.plot(
                    (x0 + 1 + i - 0.5, x0 + 1 + i - 0.5),
                    (y0, y0 + ni + 1),
                    color="gray",
                )
            # cell "bodies"
            xs = x0 + np.arange(no) + 0.5
            ys = np.array([y0 + ni + 0.5] * no)
            for x, y in zip(xs, ys):
                patches.append(
                    pltpatch.RegularPolygon(
                        (x, y - 0.4), 3, radius=0.3, orientation=0, color="#555555"
                    )
                )
            # weights
            if gray:
                colors = np.array(["black", "gray"])[(thisW.flat >= 0) + 0]
            else:
                colors = np.array(["red", "green"])[(thisW.flat >= 0) + 0]
            xs = np.arange(no) + x0 + 0.5
            ys = np.arange(ni) + y0 + 0.5
            coords = np.meshgrid(xs, ys)
            for x, y, w, c in zip(
                coords[0].flat, coords[1].flat, np.abs(thisW / maxW).flat, colors
            ):
                patches.append(
                    pltpatch.Rectangle((x - w / 2, y - w / 2), w, w, color=c)
                )
            y0 += ni + 1
            x0 += -1  # shift for next layer's constant input
        else:
            # Odd layer index. Horizontal layer. Origin is upper left.
            # Constant input
            ax.text(x0 + 0.5, y0 - 0.2, "1", fontsize=20)
            # input lines
            for i in range(ni):
                ax.plot((x0 + i + 0.5, x0 + i + 0.5), (y0, y0 + no - 0.5), color="gray")
            # output lines
            for i in range(no):
                ax.plot((x0, x0 + ni + 1), (y0 + i + 0.5, y0 + i + 0.5), color="gray")
            # cell "bodies"
            xs = np.array([x0 + ni + 0.5] * no)
            ys = y0 + 0.5 + np.arange(no)
            for x, y in zip(xs, ys):
                patches.append(
                    pltpatch.RegularPolygon(
                        (x - 0.4, y),
                        3,
                        radius=0.3,
                        orientation=-np.pi / 2,
                        color="#555555",
                    )
                )
            # weights
            if gray:
                colors = np.array(["black", "gray"])[(thisW.flat >= 0) + 0]
            else:
                colors = np.array(["red", "green"])[(thisW.flat >= 0) + 0]
            xs = np.arange(ni) + x0 + 0.5
            ys = np.arange(no) + y0 + 0.5
            coords = np.meshgrid(xs, ys)
            for x, y, w, c in zip(
                coords[0].flat, coords[1].flat, np.abs(thisW / maxW).flat, colors
            ):
                patches.append(
                    pltpatch.Rectangle((x - w / 2, y - w / 2), w, w, color=c)
                )
            x0 += ni + 1
            y0 -= 1  # shift to allow for next layer's constant input

    collection = pltcoll.PatchCollection(patches, match_original=True)
    ax.add_collection(collection)

    # Last layer output labels
    if outputNames:
        if isOdd(nLayers):
            x = x0 + 1.5
            for n in outputNames:
                x += 1
                ax.text(x, y0 + 0.5, n, fontsize=20)
        else:
            y = y0 + 0.6
            for n in outputNames:
                y += 1
                ax.text(x0 + 0.2, y, n, fontsize=20)
    ax.axis([0, xlim, ylim, 0])
    ax.axis("off")
