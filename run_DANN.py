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

from data import load_train_data, load_test_data
from dataset import SegmentDataset, SegmentTrainTestDataset

import DANN.net as net
from DANN.updater import DANNUpdater
from DANN.evaluator import DANNEvaluator


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                help='learning minibatch size')
    parser.add_argument('--optimizer', type=str, default='Adam',
                help='optimizer to use for backprop')
    parser.add_argument('--feature', '-f', type=str, default='mfcc',
                #choices=['mfcc', 'mfcc_delta', 'fbank', 'fbank_delta', 'plp', 'plp_delta'],
                help='feature type')
    parser.add_argument('--out', '-o', type=str, default='DANN/model/test',
                help='path to the output directory')
    parser.add_argument('--lam_max', type=float,
            help='lambda parameter for gradient reversal layer')
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


    enc = net.GCNN()
    lc = net.LabelClassifier(512, 6)
    dc = net.DomainClassifier(512, 1)

    if args.gpu >= 0:
        enc.to_gpu()
        lc.to_gpu()
        dc.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    if args.optimizer == 'SGD':
        opt_enc = chainer.optimizers.SGD()
        opt_lc = chainer.optimizers.SGD()
        opt_dc = chainer.optimizers.SGD()
    elif args.optimizer == 'Adam':
        opt_enc = chainer.optimizers.Adam()
        opt_lc = chainer.optimizers.Adam()
        opt_dc = chainer.optimizers.Adam()
    opt_enc.setup(enc)
    opt_lc.setup(lc)
    opt_dc.setup(dc)

    print('Preparing data...')
    dataset = SegmentTrainTestDataset(args.feature, 512, normalized=False)
    train_dat, val_dat = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.9), seed=args.seed)
    train_iter = chainer.iterators.SerialIterator(train_dat, args.batchsize, shuffle=True)
    val_iter = chainer.iterators.SerialIterator(val_dat, args.batchsize, repeat=False, shuffle=False)

    updater = DANNUpdater(
        iterators={
            'main': train_iter
        },
        models=(enc, lc, dc),
        optimizers={
            'enc': opt_enc,
            'lc': opt_lc,
            'dc': opt_dc
        },
        device=args.gpu,
        params={
            'n_epoch': args.epoch,
            'lam_max': args.lam_max
        })
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    evaluator = DANNEvaluator(
        iterators={
            'main': val_iter,
        },
        models={'enc': enc, 'lc': lc, 'dc': dc},
        device=args.gpu)
    trainer.extend(evaluator)

    snapshot_interval = (args.epoch, 'epoch')
    display_interval = (1, 'epoch')
    # snapshot
    trainer.extend(extensions.snapshot(filename='{.updater.epoch}_epoch_snapshot.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, '{.updater.epoch}.enc'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        lc, '{.updater.epoch}.lc'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dc, '{.updater.epoch}.dc'), trigger=snapshot_interval)

    # Report
    log_keys = ['epoch', 'enc/loss', 'lc/loss', 'lc/acc', 'dc/loss',
                'val/enc/loss', 'val/lc/loss', 'val/lc/acc', 'val/dc/loss', 'lam']
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=display_interval))
    trainer.extend(extensions.PrintReport(log_keys), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(
        ['enc/loss', 'val/enc/loss'], 'epoch', file_name='enc_loss.png'))
    trainer.extend(extensions.PlotReport(
        ['lc/loss', 'val/lc/loss'], 'epoch', file_name='lc_loss.png'))
    trainer.extend(extensions.PlotReport(
        ['dc/loss', 'val/dc/loss'], 'epoch', file_name='dc_loss.png'))
    trainer.run()

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--enc_path', type=str,
                help='path to the model data')
    parser.add_argument('--lc_path', type=str,
                help='path to the model data')
    parser.add_argument('--feature', '-f', type=str, default='mfcc',
                #choices=['mfcc', 'mfcc_delta', 'fbank', 'fbank_delta', 'plp', 'plp_delta'],
                help='feature type')
    parser.add_argument('--out', '-o', type=str, default='DANN/result/test',
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


    enc = net.GCNN()
    lc = net.LabelClassifier(512, 6)

    if args.gpu >= 0:
        enc.to_gpu()
        lc.to_gpu()
        print('GPU {}'.format(args.gpu))
    xp = np if args.gpu < 0 else cuda.cupy

    chainer.serializers.load_npz(args.enc_path, enc)
    print('Encoder model loaded successfully from {}'.format(args.enc_path))
    chainer.serializers.load_npz(args.lc_path, lc)
    print('Classifier model loaded successfully from {}'.format(args.lc_path))

    for kind in ['test', 'train']:
        print('Predicting with {} data...'.format(kind))
        dataset = SegmentDataset(kind, args.feature, 512, normalized=False)
        test_iter = chainer.iterators.SerialIterator(dataset, 1024, repeat=False, shuffle=False)

        posteriors = []
        labels = []
        for batch in test_iter:
            with chainer.using_config('train', False):
                x, t = convert.concat_examples(batch, args.gpu)
                post = F.softmax(lc(enc(x)))
                post = cuda.to_cpu(post.data)
                posteriors.extend(post.tolist())
                labels.extend(np.argmax(post, axis=1).tolist())

        label_list = ['MA_AD', 'MA_CH', 'MA_EL', 'FE_AD', 'FE_CH', 'FE_EL']
        if kind == 'test':
            target_data = load_test_data(args.feature)
        elif kind == 'train':
            target_data = load_train_data(args.feature)

        result = [(key, label_list[labels[i]]) for i, key in enumerate(target_data.keys())]
        with open(os.path.join(args.out, 'label_{}.tsv'.format(kind)), 'w') as wf:
            for d in result:
                wf.write('{}\t{}\n'.format(d[0], d[1]))

        result = [(key, posteriors[i]) for i, key in enumerate(target_data.keys())]
        with open(os.path.join(args.out, 'post_{}.tsv'.format(kind)), 'w') as wf:
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

