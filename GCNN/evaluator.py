import copy
import numpy as np

import chainer
from chainer import cuda, Variable
from chainer import reporter as reporter_module
import chainer.functions as F
from chainer.dataset import convert


class GCNNEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, *args, **kwargs):
        self._iterators = kwargs.pop('iterators')
        self._targets = kwargs.pop('models')
        self.device = kwargs.pop('device')

        self.converter = convert.concat_examples

        self.eval_hook = None
        self.eval_func = None

    def evaluate(self):
        iterator = self._iterators['main']
        model = self._targets['model']

        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                x, t = self.converter(batch, self.device)
                with chainer.function.no_backprop_mode():
                    y = model(x)

                    loss = F.softmax_cross_entropy(y, t)
                    acc = F.accuracy(y, t)

                    observation['val/model/loss'] = loss
                    observation['val/model/acc'] = acc

            summary.add(observation)

        return summary.compute_mean()