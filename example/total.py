#https://github.com/guoyang9/NCF/blob/master/README.md
#%%
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
# %% 
train_data = pd.read_csv('../data/train.rating', 
                         sep = '\t', 
                         header = None, 
                         names = ['user', 'item','label'],  
                         usecols = [0,1,2], 
                         dtype = {0 : np.int32, 1:np.int32, 2:np.int32})
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
                self.features_ng.append([u,j,0])
        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]
        self.features_fill =  self.features_ng + self.features_ps
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
# %%
# class NCF(nn.Module):
#     def __init__(self, num_users, num_items, embedding_dim):
#         super(NCF, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embedding_dim = embedding_dim
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#         self.embedding_user = nn.Embedding(self.num_users, 
#                                            self.embedding_dim)
#         self.embedding_item = nn.Embedding(self.num_items, 
#                                            self.embedding_dim)
#         self.fc1 = nn.Linear(self.embedding_dim*2, 
#                             self.embedding_dim//2)
#         self.fc2 = nn.Linear(self.embedding_dim//2, 
#                             self.embedding_dim//4)
#         self.fc3 = nn.Linear(self.embedding_dim//4,
#                              1)
#     def forward(self, user_indices, item_indices):
#         user_embedding = self.embedding_user(user_indices)
#         item_embedding = self.embedding_item(item_indices)
#         vector = torch.cat([user_embedding, item_embedding], dim = -1)

#         x = self.fc1(vector)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         output = self.sigmoid(x)

#         return output.squeeze()

class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MLP, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.relu = nn.ReLU()
        self.embedding_user = nn.Embedding(self.num_users, self.embedding_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim * 2, self.embedding_dim // 2)
        self.fc2 = nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4)
        self.fc3 = nn.Linear(self.embedding_dim // 4, 1)

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim = -1)
        x = self.fc1(vector)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        output = self.relu(x)

        return output

# num_user = user_num
# num_item = item_num
# embedding_dim = 64
# relu = nn.ReLU()
# embedding_user = nn.Embedding(num_user, embedding_dim)
# embedding_item = nn.Embedding(num_item, embedding_dim)
# fc1 = nn.Linear(embedding_dim * 2, embedding_dim // 2)
# fc2 = nn.Linear(embedding_dim // 2, embedding_dim // 4)
# fc3 = nn.Linear(embedding_dim // 4, 1)
# user_embedding = embedding_user(user)
# item_embedding = embedding_item(item)
# vector = torch.cat([user_embedding, item_embedding], dim = -1)
# x= fc1(vector)
# x = relu(x)
# x = fc2(x)
# x = relu(x)
# x = fc3(x)
# output_MLP = relu(x)
    
class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(GMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.embedding_user = nn.Embedding(self.num_users, self.embedding_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.embedding_dim)
        self.fc_user = nn.Linear(self.embedding_dim, 1)
        self.fc_item = nn.Linear(self.embedding_dim, 1)
    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        user = self.fc_user(user_embedding)
        item = self.fc_item(item_embedding)
        output = user * item
        return output

# num_user = user_num
# num_item = item_num
# embedding_dim = 64
# embedding_user = nn.Embedding(num_user, embedding_dim)
# embedding_item = nn.Embedding(num_item, embedding_dim)
# fc_user = nn.Linear(embedding_dim,1)
# fc_item = nn.Linear(embedding_dim,1)
# user_embedding = embedding_user(user)
# item_embedding = embedding_item(item)
# user = fc_user(user_embedding)
# item = fc_item(item_embedding)
# output = user*item
# output_GMF = output

# hap = torch.cat([output_MLP,output_GMF],dim = -1)
# linear = nn.Linear(2, 1)
# linear(hap)
#%%
# num_users = user_num
# num_items = item_num
# embed_dim = 32
# relu = nn.ReLU()
# sigmoid= nn.Sigmoid()
# embedding_user = nn.Embedding(num_users, embed_dim)
# embedding_item = nn.Embedding(num_items, embed_dim)
# fc1 = nn.Linear(64, 32)
# fc2 = nn.Linear(32, 8)
# fc3 = nn.Linear(8,1)

# user_embedding = embedding_user(user)
# item_embedding = embedding_item(item)
# vector = torch.cat([user_embedding, item_embedding], dim = -1)
# x = fc1(vector)
# x = relu(x)
# x = fc2(x)
# x = relu(x)
# output = fc3(x)
# output = sigmoid(output)
# output.squeeze()

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NeuMF,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.gmf = GMF(num_users = self.num_users,
                       num_items = self.num_items,
                       embedding_dim = self.embedding_dim)
        self.mlp = MLP(num_users = self.num_users, 
                       num_items = self.num_items, 
                       embedding_dim = self.embedding_dim)
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, user_indices, item_indices):
        gmf_vec = self.gmf(user_indices, 
                           item_indices)
        mlp_vec = self.mlp(user_indices, 
                           item_indices)
        cat = torch.cat([gmf_vec, mlp_vec], dim = -1)
        cat = self.linear(cat)
        prediction = self.sigmoid(cat)
        return prediction.squeeze()

#%%    
def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
    return np.reciprocal(np.log2(index+2))
    

def metric(model, test_loader, top_k):
    NDCG = []
    for user, item, _ in test_loader:
        user = user
        item = item
        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).numpy().tolist()
        gt_item = item[0].item()
        NDCG.append(ndcg(gt_item, recommends))
    return np.mean(NDCG)
#%%
model = NeuMF(num_users = user_num, num_items = item_num, embedding_dim= 32)
learning_rate = 0.001
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
top_k = 13

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
    NDCG = metric(model, test_loader, top_k)

    print("NDCG: {:.3f}".format(np.mean(NDCG)))

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 모델의 파라미터 수 출력
print("모델의 파라미터 수:", count_parameters(model))