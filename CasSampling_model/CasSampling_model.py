import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import argparse

import config


def get_params():

    # Training settings
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--learning_rate", type=float,
                        default='0.001', help="data directory")
    parser.add_argument("--GCN_hidden_size", type=int, default=1024)


    parser.add_argument('--MLP_hidden1', type=int, default=64)
    parser.add_argument('--MLP_hidden2', type=int, default=32)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--Activation_fc',type=str,default=nn.ReLU())
    parser.add_argument('--GCN_hidden_size2', type=int, default=8)


    args, _ = parser.parse_known_args()
    return args


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        Wh1 = torch.permute(Wh1,[1,0,2])
        Wh2 = torch.permute(Wh2,[1,0,2])
        # broadcast add
        e = Wh1 + Wh2.T
        e = torch.permute(e,[1,0,2])
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class CasSamplingNet(nn.Module):


    def __init__(self ,input_dim = 1,GCN_hidden_size=512,GCN_hidden_size2=8,MLP_hidden1=128,MLP_hidden2=32,Activation_fc = nn.ReLU()):
        super(CasSamplingNet ,self).__init__()
        self.gat1 = GraphAttentionLayer(in_features=input_dim,out_features=GCN_hidden_size,dropout=0.5,alpha=0.5,concat=True)
        self.gat2 = GraphAttentionLayer(in_features=GCN_hidden_size,out_features=GCN_hidden_size2,dropout=0.5,alpha=0.5,concat=True)
        self.FL = nn.Sequential(
            
            nn.Linear(192,MLP_hidden1),

            Activation_fc,
            # nn.Dropout(0.2),
            nn.Linear(MLP_hidden1,MLP_hidden2),
            nn.Tanh(),

            nn.Linear(MLP_hidden2,1),
            Activation_fc,
        )
        self.FL_node = nn.Sequential(
            nn.Linear(1,4),
            Activation_fc,
            nn.Linear(4,1),
            Activation_fc,
        )

        self.Weight = nn.Parameter(torch.randn(size=[128,128]))
        self.Wv = nn.Parameter(torch.randn(size=[128, 128]))
        self.rnn = nn.LSTM(1,1,num_layers=2,batch_first=True)
        self.att = nn.MultiheadAttention(embed_dim=1,num_heads=1,batch_first=True)

    def forward(self ,adjacency ,feature,time,interval_popularity):
        # #
        feature= torch.reshape(feature,[config.batch_size,128,-1])
        h = self.gat1(feature,adjacency)
        h = nn.functional.relu(h)
        h = self.gat2(h,adjacency)
        interval_popularity = torch.reshape(interval_popularity,[config.batch_size,-1,1])
        interval_popularity,_ = self.rnn(interval_popularity)
        interval_popularity = torch.reshape(interval_popularity, [config.batch_size, -1])

        h = torch.squeeze(h)


        #####
        h = torch.mean(h, dim=2, keepdim=True)

        Wh = torch.reshape(h,shape=[config.batch_size,128,-1])
        Wt = torch.reshape(time,shape=[config.batch_size,128,-1])

        attention = torch.concat([Wh, Wt], dim=2)

        attention = torch.permute(torch.matmul(torch.permute(attention,[0,2,1]),self.Weight),[0,2,1])
        attention = F.softmax(attention,dim=2)
        h = torch.reshape(h,shape=[config.batch_size,128,-1])

        time = torch.reshape(time, shape=[config.batch_size,  128,-1])

        #################
        h = torch.concat([h,time],dim=2)
        h = torch.permute(h,[0,2,1])
        h = torch.relu(torch.reshape(torch.matmul(h, self.Wv), shape=[config.batch_size, 128,-1]))
        Wh = torch.mul(h,attention)
        h = torch.sum(Wh,dim=2)

###
        h= torch.concat([h,interval_popularity],dim=1)          #(batch,lenth,dimention=1)

        h = torch.reshape(h,[config.batch_size,-1,1])
        h_att,_ = self.att(h,h,h)
        h = h_att + h

        h = torch.reshape(h,[config.batch_size,-1])

        nodes_predict = 0
        h = self.FL(h)
        return h,nodes_predict

