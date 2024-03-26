#%%
import torch
import torch.nn as nn
import pandas as pd
from preprocess import *
# %%
#%% DataLoader
dataset_path = "../dataset/Yahoo_all_data"
train_data_path: str = dataset_path + '/train.csv'
test_data_path: str = dataset_path + '/test.csv'

train_df: pd.DataFrame = pd.read_csv(train_data_path) # [0:100000]
test_df: pd.DataFrame = pd.read_csv(test_data_path)
_train_data: np.array = train_df.values.astype(np.int64)
_test_data: np.array = test_df.values.astype(np.int64)

user_positive_interaction = []
user_list: list = []
item_list: list = []

test_user_list: list = []
test_item_list: list = []
ground_truth: list = []

with open(train_data_path, 'r') as inp:
    inp.readline()
#    lines: list = inp.readlines()
    lines: list = [line.rstrip() for line in inp.readlines()]
    
    print('Begin analyze raw train file')
    pairs, user_list, item_list = analyse_interaction_from_text(lines, has_value = True)
    
    positive_pairs: list = list(filter(lambda pair: pair[2] > 0, pairs))
    
    user_positive_interaction: list = analyes_user_interacted_set(positive_pairs)
    user_positive_interaction = user_positive_interaction
    
    _train_pairs: list = pairs
    
    inp.close()
    
user_num = max(user_list + test_user_list) + 1
item_num = max(item_list + test_item_list) + 1
test_users_tensor: torch.LongTensor = torch.LongTensor(test_user_list)
test_users_tensor = test_users_tensor
sorted_ground_truth: list = [ground_truth[user_id] for user_id in test_user_list] # 사용자별 실제 구매상품

train_tensor: torch.LongTensor = torch.LongTensor(_train_data)
user_tensor = train_tensor[:,0]
item_tensor = train_tensor[:,1]
score_tensor = train_tensor[:,2]
envs_num = 2
envs: torch.LongTensor = torch.LongTensor(np.random.randint(0, envs_num, _train_data.shape[0]))


