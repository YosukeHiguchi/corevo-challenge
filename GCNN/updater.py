import numpy as np

import chainer
from chainer import cuda, Variable
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L


class GCNNUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self._iterators = kwargs.pop('iterators')
        self.model = kwargs.pop('models')
        self._optimizers = kwargs.pop('optimizers')
        self.device = kwargs.pop('device')

        self.converter = convert.concat_examples
        self.iteration = 0

    def update_core(self):
        batch = self._iterators['main'].next()
        x, t = self.converter(batch, self.device)
        x, t = Variable(x), Variable(t)
        xp = cuda.get_array_module(x.data)

        model = self.model
        opt_model = self._optimizers['model']

        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        opt_model.update()

        chainer.report({'loss': loss}, model)
        chainer.report({'acc': acc}, model)