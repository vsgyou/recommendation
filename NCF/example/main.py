#https://github.com/guoyang9/NCF/blob/master/README.md
#%%
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils.data as data
from model import *
from metric import *
from tqdm import tqdm
# %% 
train_data = pd.read_csv('../data/train.rating', 
                         sep = '\t', 
                         header = None, 
                         names = ['user', 'item'],  
                         usecols = [0,1], 
                         dtype = {0 : np.int32, 1:np.int32})
user_num = train_data['user'].max() + 1
item_num = train_data['item'].max() + 1


train_data = train_data.values.tolist()
train_mat = sp.dok_matrix((user_num, item_num), dtype = np.float32)
for x in tqdm(train_data):
    train_mat[x[0],x[1]] = 1.0
test_data = []
with open('../data/test.negative', 'r') as fd:
    line = fd.readline()
    while line != None and line != '':
        arr = line.split('\t')
        u = eval(arr[0])[0]
        test_data.append([u,eval(arr[0])[1]])
        for i in arr[1:]:
            test_data.append([u, int(i)])
        line = fd.readline()

# %%
class NCFData(data.Dataset):
    def __init__(self, features, num_item, train_mat = None, num_ng = 0, is_training = None):
        super(NCFData, self).__init__()
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]
    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u,j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u,j])
        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]
        self.features_fill =  self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng
    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)
    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label
# %%
train_dataset = NCFData(train_data, num_item= item_num, train_mat = train_mat, num_ng = 4, is_training=True)
train_dataset.ng_sample()

test_dataset = NCFData(test_data, item_num, 0, False)
train_loader = data.DataLoader(train_dataset, batch_size = 256, shuffle = True)
test_loader = data.DataLoader(test_dataset, batch_size = 99+1, shuffle = False)

#%%
model = NeuMF(num_users = user_num, num_items = item_num, embedding_dim= 64)
learning_rate = 0.001
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
top_k = 5

for epoch in range(10):
    print('#### EPOCH {} ####'.format(epoch + 1))
    model.train()
    for batch, (user, item, label) in enumerate(tqdm(train_loader)):
        user = user
        item = item
        label = label.float()
        model.zero_grad()
        prediction = model(user, item)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
    model.eval()
    HR, NDCG = metric(model, test_loader, top_k)
    print("HR : {:.3f} \tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 모델의 파라미터 수 출력
print("모델의 파라미터 수:", count_parameters(model))