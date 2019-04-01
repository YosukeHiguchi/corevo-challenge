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

def load_wav(f_name):
    '''
    Loads wav file as list
    
    Returns:
    fs: int
        sampling frequency
    
    data: list
        wav data
    '''
    try:
        fs, data = wavfile.read('corevo/raw/train0/{}.wav'.format(f_name))
    except FileNotFoundError:
        try:
            fs, data = wavfile.read('corevo/raw/train1/{}.wav'.format(f_name))
        except FileNotFoundError:
            print('file {}.wav not found'.format(f_name))
            return -1
    
    return fs, data

def apply_vad(x, seglen, r=200):
    offset = int(r / 2)

    feat = np.array(x[:, -3]) # voice active range by pitch
    feat_pad = np.pad(feat, (offset, offset), 'constant')
    mean = np.zeros_like(feat)
    diff = np.zeros_like(feat)

    for i in range(offset, feat.shape[0] + offset):
        mean[i - offset] = np.mean(feat_pad[i - offset: i + offset])

    for i in range(0, feat.shape[0] - 1):
        diff[i] = mean[i + 1] - mean[i]

    st = np.argmin(diff)

    return x[st: min(st + 512, feat.shape[0])]

def get_data_with_seglen(data_type, feature_type, seglen):
    if data_type == 'train':
        data = load_train_data(feature_type)
    elif data_type == 'test':
        data = load_test_data(feature_type)

    for key in data.keys():
        uttlen = data[key]['features'].shape[0]

        if uttlen > 512:
            data[key]['features'] = apply_vad(data[key]['features'], seglen)
            uttlen = data[key]['features'].shape[0]

        if uttlen < 512:
            data[key]['features'] = np.tile(data[key]['features'], (int(seglen / uttlen) + 1, 1))[0: 512]

    return data

if __name__ == '__main__':
    data = load_train_data('mfcc')
    # load_test_data('mfcc')

    import pdb; pdb.set_trace()
