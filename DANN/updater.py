import numpy as np

import chainer
from chainer import cuda, Variable
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L


class DANNUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self._iterators = kwargs.pop('iterators')
        self.enc, self.lc, self.dc = kwargs.pop('models')
        self._optimizers = kwargs.pop('optimizers')
        self.device = kwargs.pop('device')

        self.converter = convert.concat_examples
        self.iteration = 0

        params = kwargs.pop('params')
        self.n_epoch = params['n_epoch']
        self.lam_max = params['lam_max']

    def update_core(self):
        batch = self._iterators['main'].next()
        x, t = self.converter(batch, self.device)
        xp = cuda.get_array_module(x)

        x_train = Variable(x[xp.where(t != -1)[0]])
        t_train = Variable(t[xp.where(t != -1)[0]])
        d_train = Variable(xp.zeros(t_train.shape, dtype=xp.int32))
        x_test = Variable(x[xp.where(t == -1)[0]])
        d_test = Variable(xp.ones(x_test.shape[0], dtype=xp.int32))

        enc = self.enc
        lc = self.lc
        dc = self.dc
        opt_enc = self._optimizers['enc']
        opt_lc = self._optimizers['lc']
        opt_dc = self._optimizers['dc']

        p = float(self.epoch) / self.n_epoch
        scale = 2. / (1. + xp.exp(-10. * p, dtype=np.float32)) - 1

        z_train = enc(x_train)
        y_lc_train = lc(z_train)
        y_dc_train = dc(z_train, self.lam_max * scale)
        z_test = enc(x_test)
        y_dc_test = dc(z_test, self.lam_max * scale)

        lc_loss = F.softmax_cross_entropy(y_lc_train, t_train)
        dc_loss = F.sigmoid_cross_entropy(y_dc_train, d_train)
        dc_loss += F.sigmoid_cross_entropy(y_dc_test, d_test)
        loss = lc_loss + dc_loss

        lc_acc = F.accuracy(y_lc_train, t_train)

        enc.cleargrads()
        lc.cleargrads()
        dc.cleargrads()
        loss.backward()
        opt_enc.update()
        opt_lc.update()
        opt_dc.update()

        chainer.report({'loss': loss}, enc)
        chainer.report({'loss': lc_loss}, lc)
        chainer.report({'loss': dc_loss}, dc)
        chainer.report({'acc': lc_acc}, lc)
        chainer.report({'lam': self.lam_max * scale})