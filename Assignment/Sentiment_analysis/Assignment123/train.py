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

def train(net,model_name):

    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    optimizer = optim.Adam(net.parameters(), lr = 0.0001, weight_decay=0.0001)

    step = 0
    n_epochs = 70
    clip = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    for epoch in range(n_epochs):
        
        for inputs, labels in train_loader:
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            net.zero_grad()
            output, h = net(inputs)
            loss = criterion(output.squeeze(), labels.float())

            loss.backward()
            
            #To prevent exploding gradients
            nn.utils.clip_grad_norm(net.parameters(), clip)
            optimizer.step()
            
    #         if loss.item() < 0.02:
    #             break
            
            
            if (step % 50) == 0:            
                net.eval()
                valid_losses = []
                num_val_batch =0 
                for v_inputs, v_labels in valid_loader:
                    num_val_batch += 1
                    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

                    
                    v_output, v_h = net(v_inputs)
                    v_loss = criterion(v_output.squeeze(), v_labels.float())
                    valid_losses.append(v_loss.item())
                
                valid_losses = sum(valid_losses)/len(valid_losses)
                    
                print("Epoch: {}/{}".format((epoch+1), n_epochs),
                    "Step: {}".format(step),
                    "Training Loss: {:.4f}".format(loss.item()),
                    "Validation Loss: {:.4f}".format(valid_losses),
                    )

                if valid_losses - loss.item() > 0.2:
                    break
                    
                net.train()
    save_checkpoint(model_name,net)

if __name__ == '__main__':

        
    n_vocab=embedding_matrix.shape[0]
    n_embed=embedding_matrix.shape[1]
    n_hidden = 512
    n_hidden_decode = 512
    v_attention_dimensionality = 512
    n_output = 1   # 1 ("positive") or 0 ("negative")
    layers = 1

    #CNN
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

    train(cnn,"cnn.pt")