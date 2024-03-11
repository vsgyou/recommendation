#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# %%
movies_data = pd.read_csv("movies.dat", delimiter = '::', header = None, encoding='latin1', names = ['MovieID', 'Title', 'Genres'])
ratings_data = pd.read_csv("ratings.dat", delimiter = '::', header = None, encoding = 'latin1', names = ['UserID','MovieID', 'Rating', 'Timestamp'])
users_data = pd.read_csv("users.dat", delimiter = '::', header = None, encoding = 'latin1',names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
#%%
# train, valid 데이터 나누기
data = ratings_data
np.random.seed(3)
mask = np.random.rand(len(data)) < 0.8
train = data[mask].copy()
val = data[~mask].copy()
# #%%
# def proc_col(col, train_col = None):
#     if train_col is not None:
#         uniq = train_col.unique()
#     else:
#         uniq = col.unique()
#     name2idx = {o:i for i,o in enumerate(uniq)}
#     return name2idx, np.array([name2idx.get(x,-1) for x in col]), len(uniq)
# # %%
# def encode_data(df, train = None):
#     df = df.copy()
#     for col_name in ["UserID", "MovieID"]:
#         train_col = None
#         if train is not None:
#             train_col = train[col_name]
#         _,col,_ = proc_col(df[col_name], train_col)
#         df[col_name] = col
#         df = df[df[col_name] >= 0]
#         return df
# # %%
# df_train = encode_data(train)
# df_val = encode_data(val,train)
# embed = nn.Embedding(10,3)
# a = torch.LongTensor([[1,2,0,4,5,1]])
# embed(a)
# # %%
# class MF(nn.Module):
#     def __init__(self, num_users, num_items, emb_size = 100):
#         super(MF, self).__init__()
#         self.user_emb = nn.Embedding(num_users, emb_size)
#         self.item_emb = nn.Embedding(num_items, emb_size)
#         self.user_emb.weight.data.uniform_(0,0.05)
#         self.item_emb.weight.data.uniform_(0,0.05)

#     def forward(self,u,v):
#         u = self.user_emb(u)
#         v = self.user_emb(v)
#         return (u*v).sum(1)
# # %%
# num_users = len(df_train.UserID.unique())
# num_items = len(df_train.MovieID.unique())
# model = MF(num_users, num_items, emb_size = 100)
# def train_epocs(model, epochs = 10, lr= 0.01, wd = 0.0, unsqueeze = False):
#     optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
#     model.train()
#     for i in range(epochs):
#         users = torch.LongTensor(df_train.UserID.values)
#         items = torch.LongTemsor(df_train.MovieID.values)
#         ratings = torch.FloatTensor(df_train.rating.values)
#         if unsqueeze:
#             ratings = ratings.unsqueeze(1)
#         y_hat = model(users, items)
#         loss = F.mse_loss(y_hat, ratings)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(loss.item())
#     test_loss(model, unsqueeze)


#%%
movie = movies_data.loc[:,['MovieID','Title']]
rating = ratings_data.loc[:,['UserID', 'MovieID','Rating']]
data_merge = pd.merge(movie, rating)
data_merge = data_merge.iloc[:500000,:]
# %%
pivot_table = data_merge.pivot_table(index = ["UserID"], columns = ["Title"], values = "Rating")
#%%
# train, test set 
data_merge['Rating'] = 1
data_merge['timestamp'] = data_merge['UserID'].index
data_merge['rank_latest'] = data_merge.groupby(['UserID'])['timestamp'].rank(method = 'first', ascending = False)
train_rating = data_merge[data_merge['rank_latest']!=1]
test_ratings = data_merge[data_merge['rank_latest'] == 1]
train_rating = train_rating[['UserID', 'MovieID', 'Rating']]
test_ratings = test_ratings[['UserID','MovieID','Rating']]
#%%
# negative sampling
all_rest = data_merge['MovieID'].unique()
user, items, labels = [],[],[]
user_item_set = set(zip(train_rating['UserID'], train_rating['MovieID']))

num_negatives = 4
for (u,i) in tqdm(user_item_set):
    user.append(u)
    items.append(i)
    labels.append(1)

    for _ in range(num_negatives):
        negative_item = np.random.choice(all_rest)
        while (u, negative_item) in user_item_set:
            negative_item = np.random.choice(all_rest)
        user.append(u)
        items.append(negative_item)
        labels.append(0)
user = np.array(user)
items = np.array(items)
labels = np.array(labels)

train_df = pd.DataFrame({'UserID':user, 'MovieID':items, 'labels':labels})

test_df = test_ratings.rename({'Rating':'labels'}, axis = 1)
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
train_dataset = Rating_Dataset(train_df)
test_dataset = Rating_Dataset(test_df)
train_dataloader = DataLoader(train_dataset, batch_size = 256)
test_dataloader = DataLoader(test_dataset, batch_size = 99)
#%%
class NCF(nn.Module):
    def __init__(self, num_users, num_items):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = 16
        self.relu=nn.ReLU()
        self.sigmoid= nn.Sigmoid()
 
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim)
 
        self.fc1 = nn.Linear(in_features=32, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=1)
 
    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
 
        x = self.fc1(vector)
        x = self.relu(x)
        # x = nn.BatchNorm1d()(x)
        # x = nn.Dropout(p=0.1)(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        out = self.sigmoid(x)
 
        return out.squeeze()
#%%
#num_users = data_merge['UserID'].nunique()+1
#num_items = data_merge['MovieID'].nunique()+1
num_users = data_merge['UserID'].max() +1
num_items = data_merge['MovieID'].max() +1

model = NCF(num_users, num_items)
# %%
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0
def matrix(model, test_loader, top_k):
    HR, NDCG = [], []
    for user, item, _ in test_loader:
        user = user
        item = item

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).numpy().tolist()
        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)
# %%
learning_rate = 0.001
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
size = len(train_dataloader.dataset)
count, best_hr = 0, 0
top_k = 10
writer = SummaryWriter()

for epoch in range(10):
    print('#### EPOCH {} ####'.format(epoch + 1))
    model.train()
    start_time = time.time()
    for batch, (user, item, label) in enumerate(tqdm(train_dataloader)):
        user = user
        item = item
        label = label.float()
        model.zero_grad()
        prediction = model(user, item)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
        writer.add_scalar("data/loss",loss.item(), count)
        count += 1
    model.eval()
    HR, NDCG = matrix(model, test_dataloader, top_k)
    elapsed_time = time.time() - start_time

    print("The time elapse of epoch {:03d}".format(epoch)
          + "is:"
          + time.strftime("%H: %M: %S", time.gmtime(elapsed_time))
    )
    print("HR: {:.3f} \tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
writer.flush()
print(
    "End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
        best_epoch, best_hr, best_ndcg
    )
)
#%%