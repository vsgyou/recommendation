import torch
import torch.nn as nn 

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
        output = self.sigmoid(x)

        return output.squeeze()

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
        self.fc3 = nn.Linear(self.embedding_dim // 4, self.embedding_dim // 8)
        self.fc4 = nn.Linear(self.embedding_dim // 8, 1)

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim = -1)
        x = self.fc1(vector)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        output = x
        return output
    
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
