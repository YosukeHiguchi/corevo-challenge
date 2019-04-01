import copy
import numpy as np

import chainer
from chainer.backends import cuda
from chainer import Variable
from chainer import reporter as reporter_module
import chainer.functions as F
from chainer.dataset import convert


class DANNEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, *args, **kwargs):
        self._iterators = kwargs.pop('iterators')
        self._targets = kwargs.pop('models')
        self.device = kwargs.pop('device')

        self.converter = convert.concat_examples

        self.eval_hook = None
        self.eval_func = None

    def evaluate(self):
        iterator = self._iterators['main']
        enc = self._targets['enc']
        lc = self._targets['lc']
        dc = self._targets['dc']

        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                x, t = self.converter(batch, self.device)
                xp = cuda.get_array_module(x)

                x_train = x[xp.where(t != -1)[0]]
                t_train = t[xp.where(t != -1)[0]]
                d_train = xp.zeros(t_train.shape, dtype=xp.int32)
                x_test = x[xp.where(t == -1)[0]]
                d_test = xp.ones(x_test.shape[0], dtype=xp.int32)

                with chainer.function.no_backprop_mode():
                    z_train = enc(x_train)
                    y_lc_train = lc(z_train)
                    y_dc_train = dc(z_train, 0)
                    z_test = enc(x_test)
                    y_dc_test = dc(z_test, 0)

                    lc_loss = F.softmax_cross_entropy(y_lc_train, t_train)
                    dc_loss = F.sigmoid_cross_entropy(y_dc_train, d_train)
                    dc_loss += F.sigmoid_cross_entropy(y_dc_test, d_test)
                    loss = lc_loss + dc_loss

                    lc_acc = F.accuracy(y_lc_train, t_train)

                    observation['val/enc/loss'] = loss
                    observation['val/lc/loss'] = lc_loss
                    observation['val/dc/loss'] = dc_loss
                    observation['val/lc/acc'] = lc_acc

            summary.add(observation)

        return summary.compute_mean()

