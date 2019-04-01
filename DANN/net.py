import chainer
import chainer.functions as F
import chainer.links as L


class GCNN(chainer.Chain):
    def __init__(self, ch=4):
        super(GCNN, self).__init__()
        with self.init_scope():
            self.conv_l1 = L.Convolution2D(1, ch, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv_g1 = L.Convolution2D(1, ch, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv_l2 = L.Convolution2D(ch, ch * 2, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv_g2 = L.Convolution2D(ch, ch * 2, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv_l3 = L.Convolution2D(ch * 2, ch * 4, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv_g3 = L.Convolution2D(ch * 2, ch * 4, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv_l4 = L.Convolution2D(ch * 4, ch * 8, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv_g4 = L.Convolution2D(ch * 4, ch * 8, (4, 4), stride=(2, 2), pad=(1, 1))
            self.bn_l1 = L.BatchNormalization(ch)
            self.bn_g1 = L.BatchNormalization(ch)
            self.bn_l2 = L.BatchNormalization(ch * 2)
            self.bn_g2 = L.BatchNormalization(ch * 2)
            self.bn_l3 = L.BatchNormalization(ch * 4)
            self.bn_g3 = L.BatchNormalization(ch * 4)
            self.bn_l4 = L.BatchNormalization(ch * 8)
            self.bn_g4 = L.BatchNormalization(ch * 8)
            # self.fc = L.Linear(None, n_out)

    def __call__(self, x):

        h = F.dropout(self.bn_l1(self.conv_l1(x)) * F.sigmoid(self.bn_g1(self.conv_g1(x))), 0.2)

        h = F.dropout(self.bn_l2(self.conv_l2(h)) * F.sigmoid(self.bn_g2(self.conv_g2(h))), 0.2)

        h = F.dropout(self.bn_l3(self.conv_l3(h)) * F.sigmoid(self.bn_g3(self.conv_g3(h))), 0.2)

        # h = F.dropout(self.bn_l4(self.conv_l4(h)) * F.sigmoid(self.bn_g4(self.conv_g4(h))), 0.2)
        h = self.bn_l4(self.conv_l4(h)) * F.sigmoid(self.bn_g4(self.conv_g4(h)))

        # h = self.fc(h)

        return h

class LabelClassifier(chainer.Chain):
    def __init__(self, n_h, n_out):
        super(LabelClassifier, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_h)
            self.fc2 = L.Linear(n_h, n_out)
            self.bn1 = L.BatchNormalization(n_h)

    def __call__(self, x):
        h = F.dropout(F.relu(self.bn1(self.fc1(x))), 0.2)

        h = self.fc2(h)

        return h

class GRL(chainer.Function):
    def __init__(self, lam):
        self.lam = lam

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, gradients):
        gw, = gradients
        return -1. * self.lam * gw,

class DomainClassifier(chainer.Chain):
    def __init__(self, n_h, n_out):
        super(DomainClassifier, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_h)
            self.fc2 = L.Linear(n_h, n_out)
            self.bn1 = L.BatchNormalization(n_h)

    def __call__(self, x, lam):
        h = GRL(lam)(x)

        h = F.dropout(F.relu(self.bn1(self.fc1(h))), 0.2)

        h = self.fc2(h)

        return F.squeeze(h)