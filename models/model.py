import torch
from torch import nn
import numpy
# import sys
# sys.path.append("./")
# from Utils import args_lstm as args


class Bonds_LSTM(nn.Module):

    def __init__(self, args):
        super(Bonds_LSTM, self).__init__()
        # params
        self.args = args
        self.input_dim = args.input_dim
        self.fc1_dim = args.fc1_dim
        self.lstm_dim = args.lstm_dim
        self.fc2_dim = args.fc2_dim
        self.output_dim = args.output_dim
        self.num_lstm = args.num_lstm

        # layers
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        self.lstm = nn.LSTM(self.fc1_dim, self.lstm_dim, \
                            num_layers=self.num_lstm, batch_first=True)
        self.fc2 = nn.Linear(self.lstm_dim*self.num_lstm, self.fc2_dim)
        self.out = nn.Linear(self.fc2_dim, self.output_dim)

        # activations
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(negative_slope=1e-2)
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.activation = self.relu

    def forward(self, X, initial=False):
        Batchsize, _, _ = X.shape
        emb = self.activation(self.fc1(X))
        _, (emb, _) = self.lstm(emb)  # emb:[num_layer, batch, hidden_size]
        emb = self.fc2(self.activation(emb.reshape(Batchsize, -1)))
        output = self.activation(self.out(emb))

        return output  # output:[batch_size, output_size]


class My_Auto_Encoder(nn.Module):

    def __init__(self, args):
        super(My_Auto_Encoder, self).__init__()
        self.args = args
    
    def forward(self, X):
        return 0
