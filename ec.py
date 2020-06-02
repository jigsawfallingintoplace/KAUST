# from __future__ import print_function, division
import os
import numpy as np
import csv
import string
import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, utils
from functools import lru_cache
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



trainfile = "/home/drillthewall/KAUST/SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-train.txt"
testfile = "/home/drillthewall/KAUST/SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-dev.txt"
savepath = "/home/drillthewall/KAUST/1nn.pt"

def file_to_data(file):
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
    return data

traindata = file_to_data(trainfile)

table = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}
# new_s = s.translate(table)   

def get_all_tweets(data): 
    return [d[1].lower().translate(table) for d in data]

lexicon = []
count = dict()
all_tweets = get_all_tweets(traindata)

for tweet in all_tweets:
    for word in tweet.split():
        if word not in lexicon:
            lexicon.append(word)
            count[word] = 1
        else:
            count[word] += 1

cpy = []
for word in lexicon:
    if count[word] >= 3:
        cpy.append(word)
lexicon = cpy

def get_tweet_tensors(data):
    tweet_tensors = []
    for tweet in get_all_tweets(data)[1:]:
        tensor = torch.zeros(len(lexicon))
        for word in tweet.split():
            if word in lexicon:
                tensor[lexicon.index(word)] += 1
        tweet_tensors.append(tensor)
    return tweet_tensors

def get_label_tensors(data):
    label_tensors = []
    for d in data[1:]:
        tmp = torch.zeros(11)
        for i in range(11):
            if d[2 + i] == '1':
                tmp[i] = 1
        label_tensors.append(tmp)
    return label_tensors

tweet_tensors = get_tweet_tensors(traindata)
label_tensors = get_label_tensors(traindata)
assert len(tweet_tensors) == len(label_tensors)

train_dataset = TensorDataset(torch.stack(tweet_tensors), torch.stack(label_tensors))
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # self.f1 = nn.Linear(len(lexicon), 256)
        # self.relu = nn.ReLU()
        # self.f2 = nn.Linear(256, 11)
        self.layer_1 = nn.Linear(len(lexicon), 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 11) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        # x = nn.Softmax(dim=1)(x)
        return x

net = NN()
net.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)

net.load_state_dict(torch.load(savepath))


loss_diagram = []
num_epochs = 3
cnt = 0
running_loss = 0
for i in range(num_epochs):
    for X, Y in train_dataloader:
        cnt += 1
        l = loss_function(net(X), Y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        running_loss += l
        if (cnt % 50 == 0):
            print(l.item()/50)
            loss_diagram.append(running_loss)
            running_loss = 0

torch.save(net.state_dict(), savepath)

testdata = file_to_data(testfile)
test_tweet_tensors = get_tweet_tensors(testdata)
test_label_tensors = get_label_tensors(testdata)
assert len(test_tweet_tensors) == len(test_label_tensors)

test_dataset = TensorDataset(torch.stack(test_tweet_tensors), torch.stack(test_label_tensors))
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)

test_running_loss = 0
total_loss = 0
cnt = 0
for x, y in test_dataloader:
    cnt += 1
    guess = net(x)
    l = loss_function(guess, y)
    test_running_loss += l
    total_loss += l
    if (cnt % 10 == 0):
        print(guess[0])
        print(y[0])
        print("--------")

print("FINAL RESULT:", total_loss.item())






# traindata_raw = np.array(d)
# traindata_raw = torch.from_numpy(traindata_raw)
# tweets_traindata = traindata_raw[:, 1]