#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
test_dataloader = DataLoader(test_dataset, batch_size = 99)

for batch, (user, item, label) in enumerate(tqdm(train_dataloader)):
    user = user
    item = item
    label = label

num_user = ratings_data['UserID'].max()
num_item = ratings_data['MovieID'].max()

user_embedding = nn.Embedding(num_user, 32)
item_embeding = nn.Embedding(num_item, 32)

u = user_embedding(user)
i = item_embeding(item)
z = torch.cat([u,i],dim = 1)

fc1 = nn.Linear(64,32)
relu = nn.ReLU()
fc2 = nn.Linear(32,8)
fc3 = nn.Linear(8,1)
sigmoid = nn.Sigmoid()

x = fc1(z)
x = relu(x)
x = fc2(x)
x = relu(x)
x = fc3(x)
output = sigmoid(x)
output =output.squeeze()

output-label
