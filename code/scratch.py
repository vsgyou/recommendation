#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
# %%
ratings_data = pd.read_csv("ratings.dat", delimiter = '::', header = None, encoding = 'latin1', names = ['UserID','MovieID', 'Rating', 'Timestamp'])
ratings_data['rank_lastest'] = ratings_data.groupby('UserID')['Timestamp'].rank(method = 'first', ascending = False)
# train, test split
train_rating = ratings_data[ratings_data['rank_lastest']!=1]
test_rating = ratings_data[ratings_data['rank_lastest']==1]

train_data = train_rating[['UserID','MovieID','Rating']]
test_data = test_rating[['UserID','MovieID','Rating']]

#%%
# positive, negative data
user, item, label = [], [], []
user_item_set = set(zip(train_data['UserID'],
                        train_data['MovieID'],
                        train_data['Rating']))
for (u, i, l) in tqdm(user_item_set):
    u = u
    i = i
    l = l

    # if l<3:
    #     l = 0
    # else:
    #     l = 1
    user.append(u)
    item.append(i)
    label.append(l)
user = np.array(user)
item = np.array(item)
label = np.array(label)

train_df = pd.DataFrame({'UserID':user, 'MovieID' : item, 'labels':label})
test_df = test_data.rename({'Rating':'labels'}, axis =1)
test_df = test_df.reset_index().drop('index', axis=1)
#%%
class Rating_Dataset(torch.utils.data.Dataset):
    def __init__(self,dataset):
        self.user = dataset['UserID']
        self.item = dataset['MovieID']
        self.label = dataset['labels']
    def __len__(self):
        return len(self.user)
    def __getitem__(self,idx):
        u = self.user[idx]
        i = self.item[idx]
        l = self.label[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(l)
#%%
train_dataset = Rating_Dataset(train_df)
test_dataset = Rating_Dataset(test_df)
train_dataloader = DataLoader(train_dataset, batch_size = 256)
test_dataloader = DataLoader(test_dataset, batch_size = 99, drop_last = True)

for batch, (user, item, label) in enumerate(tqdm(train_dataloader)):
    user = user
    item = item
    label = label

num_users = ratings_data['UserID'].max()+1
num_items = ratings_data['MovieID'].max()+1


# %%
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.embedding_user = nn.Embedding(self.num_users, 
                                           self.embedding_dim)
        self.embedding_item = nn.Embedding(self.num_items, 
                                           self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim*2, 
                            self.embedding_dim//2)
        self.fc2 = nn.Linear(self.embedding_dim//2, 
                            self.embedding_dim//4)
        self.fc3 = nn.Linear(self.embedding_dim//4,
                             1)
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

        return output.squeeze()
# %%
# learning process
def loss(prediction, label):
    mse_loss = F.mse_loss(prediction, label, reduction = 'sum')
    return mse_loss

model = NCF(num_users = num_users, num_items = num_items, embedding_dim = 64)
loss_fn = nn.BCELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
for epoch in range(10):
    print('##### EPOCH {} ####'.format(epoch + 1))
    model.train()
    for batch, (user, item, label) in enumerate(tqdm(train_dataloader)):
        user = user
        item = item
        label = label.float()
        model.zero_grad()
        prediction = model(user, item)
        loss_fn = loss(prediction, label)
        loss_fn.backward()
        optimizer.step()
    model.eval()
#%%
for user, item, label in test_dataloader:
    user = user
    item = item
    label = label
    predictions = model(user, item)
    _, indices = torch.topk(predictions, 10)
    recommend = torch.take(item, indices)
    get_item = item[0].item()









#%%
train_mat = sp.dok_matrix((num_users, num_items),dtype=np.float32)
train = train_data.values.tolist()    
for x in tqdm(train):
    train_mat[x[0],x[1]] = 1.0

test_data = []
