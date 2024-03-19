#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
import optuna
#%%
# NPZ 파일 로드, 복원
train_user = np.load('../data/train_user.npz')
for i in train_user:
    print(i)

indices = train_user['indices']
indptr = train_user['indptr']
shape = train_user['shape']
data_values = train_user['data']

train_data = csr_matrix((data_values, indices, indptr), shape = shape)

#%%
num_users = train_data.shape[0]
num_items = train_data.shape[1]
embed_dim = 256
lamb = 0.01
alpha = 0.01
gamma = 0.01

user_idx = train_data.tocoo().row
user_idx = torch.tensor(user_idx)
item_idx = train_data.tocoo().col
item_idx = torch.tensor(item_idx)
label = train_data.tocoo().data
label = torch.tensor(label).float()
length = len(user_idx)


z_user_embedding = nn.Embedding(num_users, embed_dim)
c_user_embedding = nn.Embedding(num_users, embed_dim)
user_zero_vector = torch.nn.Parameter(torch.zeros(length, embed_dim), requires_grad = False)

z_item_embedding = nn.Embedding(num_items, embed_dim)
c_item_embedding = nn.Embedding(num_items, embed_dim)
item_zero_vector = torch.nn.Parameter(torch.zeros(length, embed_dim),requires_grad = False)

mlp1_weights = nn.Parameter(torch.randn(embed_dim*4, embed_dim*2))
mlp1_bias = nn.Parameter(torch.zeros(embed_dim*2))

mlp2_weights = nn.Parameter(torch.rand(embed_dim*2,1))
#%%
# forward

z_users = z_user_embedding((user_idx))
z_users_embeddings = torch.cat([z_users, user_zero_vector], dim = 1)

z_items = z_item_embedding((item_idx))
z_items_embeddings = torch.cat([z_items, item_zero_vector], dim = 1)
z = torch.cat([z_users_embeddings,z_items_embeddings], axis = 1)
z_encoded = torch.mm(z, mlp1_weights) + mlp1_bias

z_x_ij = torch.squeeze(torch.mm(torch.tanh(z_encoded), mlp2_weights))
sigmoid = nn.Sigmoid()
z_x_ij = sigmoid(z_x_ij)

c_users = c_user_embedding(user_idx)
c_users_embeddings = torch.cat([user_zero_vector, c_users], axis = 1)

c_items = c_item_embedding(item_idx)
c_items_embeddings = torch.cat([item_zero_vector, c_items], axis = 1)
c = torch.cat([c_users_embeddings, c_items_embeddings], axis = 1)

c_encoded = torch.mm(c,mlp1_weights) + mlp1_bias
c_x_ij = torch.squeeze(torch.mm(torch.tanh(c_encoded), mlp2_weights))

zc_users = z_users_embeddings + c_users_embeddings
zc_items = z_items_embeddings + c_items_embeddings
zc = torch.concat([zc_users, zc_items],dim=1)

zc_encoded = torch.mm(zc, mlp1_weights) + mlp1_bias
zc_x_ij = torch.squeeze(torch.mm(torch.tanh(zc_encoded), mlp2_weights))

# %%
mf_loss = torch.mean((1-alpha) * nn.functional.binary_cross_entropy_with_logits(z_x_ij, label) - 
                     gamma * nn.functional.binary_cross_entropy_with_logits(c_x_ij, label) + 
                     alpha * nn.functional.binary_cross_entropy_with_logits(zc_x_ij, label))

a = torch.mean(nn.functional.binary_cross_entropy_with_logits(z_x_ij, label))
b = torch.mean(nn.functional.binary_cross_entropy_with_logits(c_x_ij, label))
c = torch.mean(nn.functional.binary_cross_entropy_with_logits(zc_x_ij, label))

# unique_user_idx= torch.unique(user_idx)
# unique_users = torch.nn.functional.embedding(z_users_embeddings, unique_user_idx)
# unique_users = nn.Embedding(z_users_embeddings, unique_user_idx)
# unique_item_idx = torch.unique(item_idx)
# unique_items = torch.nn.Embedding(z_items_embeddings, unique_item_idx)
l2_loss = (torch.mean(torch.norm(z_users_embeddings, p=2)) +
           torch.mean(torch.norm(z_items_embeddings, p=2)) + 
           torch.mean(torch.norm(mlp1_weights, p=2))+
           torch.mean(torch.norm(mlp2_weights, p=2)))

loss = mf_loss + lamb * l2_loss

# %%
