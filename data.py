import os
import csv

import numpy as np
from scipy.io import loadmat, wavfile


def load_train_data(feature_type):
    data = {}

    # load labels
    with open('corevo/raw/class_train.tsv') as rf:
        reader = csv.reader(rf, delimiter='\t')
        for utt, label in reader:
            data[utt] = {}
            data[utt]['label'] = label

    # load features
    for data_name in ['train0', 'train1']:
        filepath = os.path.join('corevo/features', feature_type, data_name)
        matdata = loadmat(filepath, squeeze_me=True)

        for key in matdata.keys():
            if key[0] != '_': # is not a special attribute
                data[key[1:]]['features'] = matdata[key]['features'].item()

    return data

def load_test_data(feature_type):
    data = {}

    # load features
    filepath = os.path.join('corevo/features', feature_type, 'test')
    matdata = loadmat(filepath, squeeze_me=True)

    for key in matdata.keys():
        if key[0] != '_': # is not a special attribute
            data[key[1:]] = {}
            data[key[1:]]['features'] = matdata[key]['features'].item()

    return data

def load_wav(data_name):
    try:
        fs, data = wavfile.read('corevo/raw/train0/{}.wav'.format(data_name))
    except FileNotFoundError:
        try:
            fs, data = wavfile.read('corevo/raw/train1/{}.wav'.format(data_name))
        except FileNotFoundError:
            print('file {}.wav not found'.format(f_name))
            return -1
    
    return np.array(data)

if __name__ == '__main__':
    data = load_train_data('mfcc')
    # load_test_data('mfcc')

    import pdb; pdb.set_trace()