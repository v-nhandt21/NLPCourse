import numpy as np
import torch
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim


Vocab = []

def Norm(text):
    return text

def pad_text(encoded_reviews, seq_length):

    reviews = []

    for review in encoded_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append([0]*(seq_length-len(review)) + review)

    return np.array(reviews)

def LoadData(file, Vocab=Vocab):
    with open(file, "r",encoding="ISO-8859-1") as f:
        data_x = []
        data_y = []
        contents = f.read().splitlines()
        for line in contents:
            try:
                _,text,label = line.split("#")
            except:
                continue
            text = text.split(" ",1)[1]
            text = text.lower()
            text = "".join([ch for ch in text if ch not in punctuation])

            data_x.append(text)
            data_y.append(label)
            Vocab = Vocab + text.split(" ")
        return data_x, data_y, Vocab


train_x, train_y, Vocab = LoadData("../data/train.txt",Vocab)
dev_x, dev_y, Vocab = LoadData("../data/dev.txt",Vocab)
test_x, test_y, Vocab = LoadData("../data/test.txt",Vocab)


word_counts = Counter(Vocab)
word_list = sorted(word_counts, key = word_counts.get, reverse = True)
vocab_to_int = {word:idx+1 for idx, word in enumerate(word_list)}
int_to_vocab = {idx:word for word, idx in vocab_to_int.items()}

encoded_train = [[vocab_to_int[word] for word in review.split(" ")] for review in train_x]
train_x = pad_text(encoded_train, seq_length = 200)
train_y = np.array([1 if label == "pos" else 0 for label in train_y])

encoded_dev = [[vocab_to_int[word] for word in review.split(" ")] for review in dev_x]
dev_x = pad_text(encoded_dev, seq_length = 200)
dev_y = np.array([1 if label == "pos" else 0 for label in dev_y])

encoded_test = [[vocab_to_int[word] for word in review.split(" ")] for review in test_x]
test_x = pad_text(encoded_test, seq_length = 200)
test_y = np.array([1 if label == "pos" else 0 for label in test_y])


train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

class LSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p = 0.5):
        super().__init__()
        self.n_vocab = n_vocab     # number of unique words in vocabulary
        self.n_layers = n_layers   # number of LSTM layers 
        self.n_hidden = n_hidden   # number of hidden nodes in LSTM
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first = True, dropout = drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words):
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)         # (batch_size, seq_length, n_hidden)
        
        out = self.fc(lstm_out[:, -1, :])
        
        sig = self.sigmoid(out)
        return sig, h 


n_vocab = len(vocab_to_int)
n_embed = 400
n_hidden = 512
n_output = 1   # 1 ("positive") or 0 ("negative")
n_layers = 2

net = LSTM(n_vocab, n_embed, n_hidden, n_output, n_layers).cuda()

net.load_state_dict(torch.load("LSTM.pt"))

count = 0
device = "cuda"
net.eval()
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = net(inputs)
    #output = [ int(e) for e in torch.round(output.squeeze()).tolist() ]

    output = torch.round(output.squeeze()).detach().cpu().numpy().astype(int)
    #print(len(output))

    ground = labels.detach().cpu().numpy().astype(int)
    #print(len(ground))
    #print(np.sum(output == ground))
    count = count + np.sum(output == ground)
print(count/len(test_x))
