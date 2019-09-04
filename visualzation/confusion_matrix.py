# -*- coding: utf-8 -*-
"""
@File    : confusion_matrix.py
@Time    : 2019/9/4 11:22
@Author  : KeyForce
@Email   : july.master@outlook.com
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix

# classes_logit = [0, 1, 2]
# classes = ['normal', 'benign', 'suspicious']
# y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
# y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
# matrix = confusion_matrix(y_actu, y_pred)


# print(matrix)
# plt.imshow(matrix, interpolation='nearest', cmap='Oranges')
# plt.title('confusion_matrix')
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes)
# plt.yticks(tick_marks, classes)
#
# plt.tight_layout()
# plt.show()


def plotCM(classes, matrix):
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    plt.show()


# plotCM(classes, matrix)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap='Oranges',
                          normalize=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.show()







