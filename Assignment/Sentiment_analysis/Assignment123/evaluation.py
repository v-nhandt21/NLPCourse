
import numpy as np
import torch
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torchinfo import summary
import pandas as pd
import tensorflow as tf

from loaddata import GetData
from cnn import CNN
from rnn import NetworkGRU_reimplement, NetworkLSTM_reimplement, NetworkRNN_reimplement
from attention import AttentionLSTM_Additive, AttentionLSTM_Multiplicative, AttentionLSTM_Dot, AttentionBiLSTM_Additive
from utils import load_checkpoint, save_checkpoint
device="cuda"
train_loader, valid_loader, test_loader, embedding_matrix = GetData()

def evaluation(net,model_name):

    load_checkpoint(model_name,net)

    net.eval().to(device)
    count = 0
    sums = 0

    for v_inputs, v_labels in test_loader:
        
        sums = sums + len(v_inputs)
        
        v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

        v_output, v_h = net(v_inputs)
            

        output = torch.round(v_output.squeeze()).detach().cpu().numpy().astype(int)

        ground = v_labels.detach().cpu().numpy().astype(int)

        count = count + np.sum(output == ground)
        
    print("Accuracy: " + str(count/sums))

if __name__ == '__main__':
    n_vocab=embedding_matrix.shape[0]
    n_embed=embedding_matrix.shape[1]
    n_hidden = 512
    n_hidden_decode = 512
    v_attention_dimensionality = 512
    n_output = 1   # 1 ("positive") or 0 ("negative")
    layers = 1

    #CNN: Accuracy: 0.7845382963493199
    cnn = CNN(n_vocab, n_embed, n_hidden, n_output, layers,embedding_matrix).cuda()

    #RNN
    rnn = NetworkRNN_reimplement(n_vocab, n_embed, n_hidden, n_output, layers).cuda()

    #LSTM
    lstm = NetworkLSTM_reimplement(n_vocab, n_embed, n_hidden, n_output, layers).cuda()

    #GRU
    gru = NetworkGRU_reimplement( n_vocab, n_embed, n_hidden, n_output, layers).cuda()

    #Attention_Dot
    attention1 = AttentionLSTM_Dot(n_vocab, n_embed, n_hidden, n_output, layers).cuda()

    #Attention_Mul
    attention2 = AttentionLSTM_Multiplicative(n_vocab, n_embed, n_hidden, n_hidden_decode, n_output, layers).cuda()

    #Attention_Add
    attention3 = AttentionLSTM_Additive(n_vocab, n_embed, n_hidden, n_hidden_decode, n_output, layers, v_attention_dimensionality).cuda()

    #Attention_Bilstm_Add
    attention4 = AttentionBiLSTM_Additive(n_vocab, n_embed, n_hidden, n_hidden_decode, n_output, layers, v_attention_dimensionality).cuda()

    evaluation(rnn,"rnn.pt")