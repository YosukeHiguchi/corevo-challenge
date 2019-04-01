import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import cuda, training
from chainer.training import extensions
import chainer.functions as F
from chainer.dataset import convert

from data import load_test_data
from dataset import SegmentDataset

import GCNN.net as net
from GCNN.updater import GCNNUpdater
from GCNN.evaluator import GCNNEvaluator


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                help='learning minibatch size')
    parser.add_argument('--optimizer', type=str, default='Adam',
                help='optimizer to use for backprop')
    parser.add_argument('--feature', '-f', type=str, default='mfcc',
                #choices=['mfcc', 'mfcc_delta', 'fbank', 'fbank_delta', 'plp', 'plp_delta'],
                help='feature type')
    parser.add_argument('--out', '-o', type=str, default='GCNN/model/test',
                help='path to the output directory')
    parser.add_argument('--seed', type=int, default=0,
                help='random seed')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # Set seed
    np.random.seed(args.seed)
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        cuda.cupy.random.seed(args.seed)

    print(args)


    model = net.GCNN(6)

    if args.gpu >= 0:
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    opt_model = chainer.optimizers.Adam()
    opt_model.setup(model)

    print('Preparing data...')
    dataset = SegmentDataset('train', args.feature, 512, normalized=True)
    train_dat, val_dat = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.9), seed=args.seed)
    train_iter = chainer.iterators.SerialIterator(train_dat, args.batchsize, shuffle=True)
    val_iter = chainer.iterators.SerialIterator(val_dat, args.batchsize, repeat=False, shuffle=False)

    updater = GCNNUpdater(
        iterators={
            'main': train_iter
        },
        models=(model),
        optimizers={
            'model': opt_model
        },
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    evaluator = GCNNEvaluator(
        iterators={
            'main': val_iter,
        },
        models={'model': model},
        device=args.gpu)
    trainer.extend(evaluator)

    snapshot_interval = (args.epoch, 'epoch')
    display_interval = (1, 'epoch')
    # snapshot
    trainer.extend(extensions.snapshot(filename='{.updater.epoch}_epoch_snapshot.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, '{.updater.epoch}.model'), trigger=snapshot_interval)

    # Report
    log_keys = ['epoch', 'model/loss', 'model/acc', 'val/model/loss', 'val/model/acc']
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=display_interval))
    trainer.extend(extensions.PrintReport(log_keys), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(
        ['model/loss', 'val/model/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['model/acc', 'val/model/acc'], 'epoch', file_name='acc.png'))
    trainer.run()

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_path', type=str,
                help='path to the model data')
    parser.add_argument('--feature', '-f', type=str, default='mfcc',
                #choices=['mfcc', 'mfcc_delta', 'fbank', 'fbank_delta', 'plp', 'plp_delta'],
                help='feature type')
    parser.add_argument('--out', '-o', type=str, default='GCNN/result/test',
                help='path to the output directory')
    parser.add_argument('--seed', type=int, default=0,
                help='random seed')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # Set seed
    np.random.seed(args.seed)
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        chainer.cuda.cupy.random.seed(args.seed)

    print(args)


    model = net.GCNN(6)

    if args.gpu >= 0:
        model.to_gpu()
        print('GPU {}'.format(args.gpu))
    xp = np if args.gpu < 0 else cuda.cupy

    chainer.serializers.load_npz(args.model_path, model)
    print('Encoder model loaded successfully from {}'.format(args.model_path))

    print('Preparing data...')
    dataset = SegmentDataset('test', args.feature, 512, normalized=True)
    test_iter = chainer.iterators.SerialIterator(dataset, 1024, repeat=False, shuffle=False)

    predicted_label = []
    for batch in test_iter:
        with chainer.using_config('train', False):
            x, t = convert.concat_examples(batch, args.gpu)
            post = F.softmax(model(x))
            post = cuda.to_cpu(post.data)
            predicted_label.extend(np.argmax(post, axis=1).tolist())

    label_list = ['MA_AD', 'MA_CH', 'MA_EL', 'FE_AD', 'FE_CH', 'FE_EL']
    test_data = load_test_data(args.feature)
    result = [(key, label_list[predicted_label[i]]) for i, key in enumerate(test_data.keys())]
    with open(os.path.join(args.out, 'result.tsv'), 'w') as wf:
        for d in result:
            wf.write('{}\t{}\n'.format(d[0], d[1]))


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    cmd = sys.argv[1]
    if cmd == '--train':
        train()
    elif cmd == '--predict':
        predict()

