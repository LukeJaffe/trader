#!/usr/bin/env python3

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np

import os
import sys
import argparse

from utils import progress_bar
from torch.autograd import Variable

from dataset import DummyDataset
from wiki import WikiDataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--num_days', default=100, type=int, help='')
parser.add_argument('--dataset', default='wiki', choices=['wiki', 'dummy'], help='')
args = parser.parse_args()

# Data
print('==> Preparing data..')

if args.dataset == 'dummy':
    trainset = DummyDataset(train=True, num_days=args.num_days)
    testset = DummyDataset(train=False, num_days=args.num_days)
elif args.dataset == 'wiki':    
    trainset = WikiDataset(train=True)
    testset = WikiDataset(train=False, train_mean=trainset.train_mean, train_std=trainset.train_std)
else:
    raise Exception('Invalid dataset name: {}'.format(args.dataset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

class Sequence(nn.Module):
    def __init__(self, num_hidden=51):
        super(Sequence, self).__init__()
        self.num_hidden = num_hidden
        self.lstm1 = nn.LSTMCell(1, self.num_hidden)
        self.lstm2 = nn.LSTMCell(self.num_hidden, 1)

    def forward(self, input):
        h_t = Variable(torch.zeros(input.size(0), self.num_hidden).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.num_hidden).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
        h_t, c_t = self.lstm1(c_t2, (h_t, c_t))
        h_t2, output = self.lstm2(c_t, (h_t2, c_t2))

        return output

class Sequence2(nn.Module):
    def __init__(self, num_hidden1=51, num_hidden2=51):
        super(Sequence2, self).__init__()
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.lstm1 = nn.LSTMCell(1, self.num_hidden1)
        self.lstm2 = nn.LSTMCell(self.num_hidden1, self.num_hidden2)
        self.lstm3 = nn.LSTMCell(self.num_hidden2, self.num_hidden1)
        self.lstm4 = nn.LSTMCell(self.num_hidden1, 1)

    def forward(self, input):
        h_t = Variable(torch.zeros(input.size(0), self.num_hidden1).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.num_hidden1).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), self.num_hidden2).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), self.num_hidden2).double(), requires_grad=False)
        h_t3 = Variable(torch.zeros(input.size(0), self.num_hidden1).double(), requires_grad=False)
        c_t3 = Variable(torch.zeros(input.size(0), self.num_hidden1).double(), requires_grad=False)
        h_t4 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)
        c_t4 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(c_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(c_t3, (h_t4, c_t4))
        h_t, c_t = self.lstm1(c_t4, (h_t, c_t))
        h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
        h_t3, c_t3 = self.lstm3(c_t2, (h_t3, c_t3))
        h_t4, output = self.lstm4(c_t3, (h_t4, c_t4))

        return output

class Model(nn.Module):
    def __init__(self, num_seq=100, num_hidden=51, num_layers=2):
        super(Model, self).__init__()
        self.num_seq = num_seq
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        # Network
        self.affine = nn.Sequential(nn.Linear(self.num_seq, self.num_seq), nn.ReLU())
        self.rnn1 = nn.LSTM(1, self.num_hidden, self.num_layers, batch_first=True)
        self.rnn2 = nn.LSTM(self.num_hidden, 1, 1, batch_first=True)
        self.predictor = nn.Linear(self.num_seq, 1)

    def forward(self, input):
        h1 = Variable(torch.zeros(self.num_layers, input.size(0), 
            self.num_hidden).double(), 
            requires_grad=False)
        c1 = Variable(torch.zeros(self.num_layers, input.size(0), 
            self.num_hidden).double(), 
            requires_grad=False)
        h2 = Variable(torch.zeros(self.num_layers, input.size(0), 1).double(), 
            requires_grad=False)
        c2 = Variable(torch.zeros(self.num_layers, input.size(0), 1).double(), 
            requires_grad=False)

        a_out = self.affine(input.squeeze())
        out1, (h_n1, c_n1) = self.rnn1(a_out.unsqueeze(2), (h1, c1))
        out2, (h_n2, c_n2) = self.rnn2(out1, (h2, c2))
        output = self.predictor(out2.squeeze())
        #output = out2[:, 0, :].squeeze()

        return output

class Model2(nn.Module):
    def __init__(self, num_seq=100, num_hidden=51, num_layers=2):
        super(Model2, self).__init__()
        self.num_seq = num_seq
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        # Network
        self.affine = nn.Sequential(nn.Linear(self.num_seq, self.num_seq), nn.ReLU())
        self.rnn1 = nn.GRU(1, self.num_hidden, self.num_layers, batch_first=True)
        self.rnn2 = nn.GRU(self.num_hidden, 1, 1, batch_first=True)
        self.predictor = nn.Linear(self.num_seq, 1)

    def forward(self, input):
        h1 = Variable(torch.zeros(self.num_layers, input.size(0), 
            self.num_hidden).double(), 
            requires_grad=False)
        h2 = Variable(torch.zeros(self.num_layers, input.size(0), 1).double(), 
            requires_grad=False)

        a_out = self.affine(input.squeeze())
        out1, h_n1 = self.rnn1(a_out.unsqueeze(2), h1)
        out2, h_n2 = self.rnn2(out1, h2)
        output = self.predictor(out2.squeeze())
        #output = out2[:, 0, :].squeeze()

        return output

class Model3(nn.Module):
    def __init__(self, num_seq=100, num_hidden=51, num_layers=2):
        super(Model3, self).__init__()
        self.num_seq = num_seq
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        # Network
        self.affine = nn.Sequential(nn.Linear(self.num_seq, self.num_seq), nn.ReLU())
        self.rnn1 = nn.RNN(1, self.num_hidden, self.num_layers, batch_first=True)
        self.rnn2 = nn.RNN(self.num_hidden, 1, 1, batch_first=True)
        self.predictor = nn.Linear(self.num_seq, 1)

    def forward(self, input):
        h1 = Variable(torch.zeros(self.num_layers, input.size(0), 
            self.num_hidden).double(), 
            requires_grad=False)
        h2 = Variable(torch.zeros(self.num_layers, input.size(0), 1).double(), 
            requires_grad=False)

        a_out = self.affine(input.squeeze())
        out1, h_n1 = self.rnn1(a_out.unsqueeze(2), h1)
        out2, h_n2 = self.rnn2(out1, h2)
        output = self.predictor(out2.squeeze())
        #output = out2[:, 0, :].squeeze()

        return output

class ConvModel(nn.Module):
    def __init__(self, num_seq=100, num_hidden=200):
        super(ConvModel, self).__init__()

        self.num_seq = num_seq
        self.num_hidden = num_hidden

        self.affine1 = nn.Sequential(nn.Linear(self.num_seq, self.num_hidden), nn.BatchNorm1d(self.num_hidden), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Conv1d(1, 2, 5, stride=3), nn.BatchNorm1d(2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Conv1d(2, 4, 5, stride=3), nn.BatchNorm1d(4), nn.Tanh()) 
        self.conv3 = nn.Sequential(nn.Conv1d(4, 8, 5, stride=3), nn.BatchNorm1d(8), nn.Tanh()) 
        self.conv4 = nn.Sequential(nn.Conv1d(8, 16, 5, stride=3), nn.BatchNorm1d(16), nn.Tanh()) 
        self.pool1 = nn.Sequential(nn.AvgPool1d(2, stride=2), nn.BatchNorm1d(1), nn.Tanh())
        self.pool2 = nn.Sequential(nn.AvgPool1d(2, stride=2), nn.BatchNorm1d(1), nn.Tanh())
        self.pool3 = nn.Sequential(nn.AvgPool1d(2, stride=2), nn.BatchNorm1d(1), nn.Tanh())
        self.pool4 = nn.Sequential(nn.AvgPool1d(2, stride=2))

    def forward(self, input):
        out1 = self.affine1(input.squeeze())
        out2 = self.conv1(out1.unsqueeze(1))
        out3 = self.conv2(out2)
        out4 = self.conv3(out3)
        out5 = self.conv4(out4).squeeze().unsqueeze(1)
        out6 = self.pool1(out5)
        out7 = self.pool2(out6)
        out8 = self.pool3(out7)
        out9 = self.pool4(out8)
        return out9

class ConvModel2(nn.Module):
    def __init__(self, num_seq=100, num_hidden=200):
        super(ConvModel2, self).__init__()

        self.num_seq = num_seq
        self.num_hidden = num_hidden

        self.affine1 = nn.Sequential(nn.Linear(self.num_seq, self.num_hidden), nn.BatchNorm1d(self.num_hidden), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Conv1d(1, 2, 5, stride=3), nn.BatchNorm1d(2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Conv1d(2, 4, 5, stride=3), nn.BatchNorm1d(4), nn.Tanh()) 
        self.conv3 = nn.Sequential(nn.Conv1d(4, 8, 5, stride=3), nn.BatchNorm1d(8), nn.Tanh()) 
        self.conv4 = nn.Sequential(nn.Conv1d(8, 16, 5, stride=3), nn.BatchNorm1d(16), nn.Tanh()) 
        self.conv5 = nn.Sequential(nn.Conv1d(16, 32, 5, stride=3), nn.BatchNorm1d(32), nn.Tanh()) 
        self.pool1 = nn.Sequential(nn.AvgPool1d(2, stride=2), nn.BatchNorm1d(1), nn.Tanh())
        self.pool2 = nn.Sequential(nn.AvgPool1d(2, stride=2), nn.BatchNorm1d(1), nn.Tanh())
        self.pool3 = nn.Sequential(nn.AvgPool1d(2, stride=2), nn.BatchNorm1d(1), nn.Tanh())
        self.pool4 = nn.Sequential(nn.AvgPool1d(2, stride=2), nn.BatchNorm1d(1), nn.Tanh())
        self.pool5 = nn.Sequential(nn.AvgPool1d(2, stride=2))

    def forward(self, input):
        out1 = self.affine1(input.squeeze())
        out2 = self.conv1(out1.unsqueeze(1))
        out3 = self.conv2(out2)
        out4 = self.conv3(out3)
        out5 = self.conv4(out4)
        out6 = self.conv5(out5)
        out7 = self.pool1(out6.squeeze().unsqueeze(1))
        out8 = self.pool2(out7)
        out9 = self.pool3(out8)
        out10 = self.pool4(out9)
        out11 = self.pool5(out10)
        #print(out11.size(), type(out11.data))
        #sys.exit()
        return out11

class MLP(nn.Module):
    def __init__(self, num_seq=100, num_hidden=200, num_layers=2):
        super(MLP, self).__init__()
        
        self.num_seq = num_seq
        self.num_hidden = num_hidden

        layers = [nn.Sequential(nn.Linear(self.num_seq, self.num_hidden), nn.BatchNorm1d(self.num_hidden), nn.Tanh())]
        for i in range(num_layers):
            layers.append(nn.Sequential(nn.Linear(self.num_hidden, self.num_hidden), nn.BatchNorm1d(self.num_hidden), nn.Tanh()))
        layers.append(nn.Linear(self.num_hidden, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        output = self.net(input.squeeze())
        return output

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(1536, 1)

    def forward(self, x):
        out = self.features(x.transpose(1,2))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm1d(x),
                           nn.Tanh()]
                in_channels = x
        layers += [nn.AvgPool1d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class RNNModel(nn.Module):
    def __init__(self, num_seq=100, num_hidden=51, num_layers=2):
        super(RNNModel, self).__init__()
        self.num_seq = num_seq
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        # Network
        self.affine = nn.Sequential(nn.Linear(self.num_seq, self.num_seq), nn.ReLU())
        self.rnn = nn.GRU(1, self.num_hidden, self.num_layers, batch_first=True)
        conv_layers = []
        for i in range(5):
            conv_layers += [nn.Conv1d(self.num_seq, self.num_seq, kernel_size=3, stride=3),
                           nn.BatchNorm1d(self.num_seq),
                           nn.ReLU()]
        self.conv = nn.Sequential(*conv_layers)
        self.predictor = nn.Linear(self.num_seq, 1)

    def forward(self, input):
        h = Variable(torch.zeros(self.num_layers, input.size(0), 
            self.num_hidden).double(), 
            requires_grad=False)

        a_out = self.affine(input.squeeze())
        rnn_out, rnn_h = self.rnn(a_out.unsqueeze(2), h)
        conv_out = self.conv(rnn_out)
        output = self.predictor(conv_out.squeeze())

        return output

print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = DPN92()
# net = ShuffleNetG2()
#net = Model(num_hidden=51, num_layers=2)
#net = Model(num_hidden=101, num_layers=3)
#net = Model2(num_hidden=51, num_layers=2)
#net = Model3(num_hidden=51, num_layers=2)
#net = ConvModel(num_hidden=200)
#net = ConvModel2(num_hidden=500)
#net = MLP(num_hidden=200, num_layers=10)
#net = VGG('VGG19')
net = RNNModel(num_hidden=256)
net.double()

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #if use_cuda:
        #    inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.unsqueeze(2)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        outdata = outputs.data
        outdata[outdata > 0] = 1
        outdata[outdata <= 0] = -1
        predicted = outdata.long()
        total += targets.size(0)
        correct += predicted.eq(targets.data.long()).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

inputs, targets, outputs = None, None, None
def train_lbfgs(epoch):
    global inputs, targets, outputs
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #if use_cuda:
        #    inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.unsqueeze(2)
        inputs, targets = Variable(inputs), Variable(targets)
        def closure():
            global inputs, targets, outputs
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss
        loss = optimizer.step(closure)

        try:
            train_loss += loss.data[0]
        except:
            raise
        outdata = outputs.data
        outdata[outdata > 0] = 1
        outdata[outdata <= 0] = -1
        predicted = outdata.long()
        total += targets.size(0)
        correct += predicted.eq(targets.data.long()).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        #if use_cuda:
        #    inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.unsqueeze(2)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        outdata = outputs.data
        outdata[outdata > 0] = 1
        outdata[outdata <= 0] = -1
        predicted = outdata.long()
        total += targets.size(0)
        correct += predicted.eq(targets.data.long()).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_simple1(num_avg=1):
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        predicted = []
        for d in inputs:
            if d[-num_avg:].mean() > d[0:num_avg].mean():
                predicted.append(-1)
            else:
                predicted.append(1) 
        predicted = torch.LongTensor(np.array(predicted))

        total += targets.size(0)
        correct += predicted.eq(targets.long()).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_simple2(window_len=20):
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        predicted = []
        for d in inputs:
            counter = 0
            for i in range(len(d)-window_len):
                if d[i] <= d[i+window_len]:
                    counter += 1
                else:
                    counter -= 1
            if counter > 0:
                predicted.append(-1)
            else:
                predicted.append(1)

        predicted = torch.LongTensor(np.array(predicted))

        total += targets.size(0)
        correct += predicted.eq(targets.long()).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#for i in range(1, 100):
#    test_simple2(window_len=i)
#sys.exit()


epoch = 0
train_loss = 100
while train_loss >= 0.05:
    train_loss = train(epoch)
    epoch += 1
test(epoch)
