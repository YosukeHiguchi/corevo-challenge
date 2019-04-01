import sys
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

from data import load_train_data

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    file_path = sys.argv[1]

    train_data = load_train_data('mfcc')
    predict = {}
    with open(file_path) as rf:
        reader = csv.reader(rf, delimiter='\t')
        for utt, label in reader:
            predict[utt] = {}
            predict[utt]['label'] = label

    y_true = []
    y_pred = []
    for key in train_data.keys():
        y_true.append(train_data[key]['label'])
        y_pred.append(predict[key]['label'])

        # if train_data[key]['label'] != predict[key]['label']:
        #     print('{} {} {}'.format(key, train_data[key]['label'], predict[key]['label']))


    conf_mat = confusion_matrix(y_true, y_pred, labels=['MA_AD', 'MA_CH', 'MA_EL', 'FE_AD', 'FE_CH', 'FE_EL'])
    plot_confusion_matrix(conf_mat, classes=['MA_AD', 'MA_CH', 'MA_EL', 'FE_AD', 'FE_CH', 'FE_EL'], normalize=False)
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.savefig('/'.join(file_path.split('/')[0:-1]) + '/cm.png')

if __name__ == '__main__':
    main()