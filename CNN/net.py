import chainer
import chainer.functions as F
import chainer.links as L


class CNN(chainer.Chain):
    def __init__(self, n_out, ch=4):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, ch, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv2 = L.Convolution2D(ch, ch * 2, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv3 = L.Convolution2D(ch * 2, ch * 4, (4, 4), stride=(2, 2), pad=(1, 1))
            self.conv4 = L.Convolution2D(ch * 4, ch * 8, (4, 4), stride=(2, 2), pad=(1, 1))
            self.bn1 = L.BatchNormalization(ch)
            self.bn2 = L.BatchNormalization(ch * 2)
            self.bn3 = L.BatchNormalization(ch * 4)
            self.bn4 = L.BatchNormalization(ch * 8)
            self.fc = L.Linear(None, n_out)

    def __call__(self, x):

        h = F.dropout(F.relu(self.bn1(self.conv1(x))), 0.2)

        h = F.dropout(F.relu(self.bn2(self.conv2(h))), 0.2)

        h = F.dropout(F.relu(self.bn3(self.conv3(h))), 0.2)

        h = F.dropout(F.relu(self.bn4(self.conv4(h))), 0.2)

        h = self.fc(h)

        return h