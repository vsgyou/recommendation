#%%
import os
import math
import json
import wandb
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from torch.autograd import Function

from model import InvPrefImplicit, _init_eps
from preprocess import *
#%%            
def cluster(model, 
            envs, 
            batch_size, 
            users_tensor, 
            items_tensor, 
            scores_tensor, 
            envs_num, 
            const_env_tensor_list, 
            cluster_distance_func, 
            eps_random_tensor) -> int:
    cluster_use_random_sort = False
    model.eval()
    new_env_tensors_list: list = []
    
    for (batch_index, 
         (batch_users_tensor, 
          batch_items_tensor, 
          batch_scores_tensor)) in enumerate(mini_batch(batch_size, 
                                                        users_tensor, 
                                                        items_tensor, 
                                                        scores_tensor)):
        distances_list: list = []
        for env_idx in range(envs_num):
            envs_tensor: torch.Tensor = const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]
            print('envs_tensor:', envs_tensor.shape, envs_tensor)
            cluster_pred: torch.Tensor = model.cluster_predict(batch_users_tensor, batch_items_tensor, envs_tensor)
            print('cluster_pred:', cluster_pred)
            distances: torch.Tensor = cluster_distance_func(cluster_pred, batch_scores_tensor)
            print('distances:', distances)
            distances = distances.reshape(-1,1)
            print('distances reshape:', distances)
            distances_list.append(distances)
            
        each_envs_distances : torch.Tensor = torch.cat(distances_list, dim = 1)
        
        if cluster_use_random_sort:
            sort_random_index: np.array = np.random.randint(1, eps_random_tensor.shape[0], each_envs_distances.shape[1])
            random_eps: torch.Tensor = eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps
            
        new_envs: torch.Tensor = torch.argmin(each_envs_distances, dim = 1)
        new_env_tensors_list.append(new_envs)
    all_new_env_tensors: torch.Tensor = torch.cat(new_env_tensors_list, dim = 0)
    envs_diff: torch.Tensro = (envs - all_new_env_tensors) != 0
    diff_num: int = int(torch.sum(envs_diff))
    
    return all_new_env_tensors, diff_num

#%%
# model_config
env_num = 2
factor_num = 40
reg_only_embed = True
reg_env_embed = False

# train_config
batch_size = 8192
epochs = 100
cluster_interval = 5
evaluate_interval = 10
lr = 0.005
invariant_coe = 3.351991776096847
env_aware_coe = 9.988658447411407
env_coe = 9.06447753571379
L2_coe = 3.1351402017943117
L1_coe = 0.4935216278026648
alpha = 1.9053711444718746
use_class_re_weight = True
use_recommend_re_weight = False
test_begin_epoch = 0
begin_cluster_epoch = None
stop_cluster_epoch = None


# evaluate_config
top_k_list = [3, 5, 7]
test_batch_size = 1024
eval_k = 5
eval_metric = "ndcg"

RANDOM_SEED_LIST = [17373331, 17373511, 17373423]
DATASET_PATH = '/Yahoo_all_data/'
METRIC_LIST = ['ndcg', 'recall', 'precision']

# expt config
random_seed = 0
has_item_pool: bool = False
cluster_use_random_sort: bool = False # org True

#%% WANDB
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
save_dir = f"./weights/expt_{expt_num}"

"""WandB"""
wandb_var = wandb.init(project = "drs",
                       config = {
                           "device" : "cpu",
                           "env_num" : 2,
                           "facor_num" : 40,
                           "reg_only_embed" : True,
                           "reg_env_embed" : False,
                           "batch_size" : 8192,
                           "epochs" : 1000,
                           "cluster_interval" : 5,
                           "evaluate_interval" : 10,
                           "lr" : 0.005,
                           "invariant_coe" : 3.351991776096847,
                           "env_aware_coe" : 9.988658447411407,
                           "env_coe" : 9.06447753571379,
                           "L2_coe" : 3.1351402017943117,
                           "L1_coe" : 0.4935216278026648,
                           "alpha" : 1.9053711444718746,
                           "use_class_re_weight" : True,
                           "use_recommend_re_weight" : False,
                           "test_begin_epoch" : 0,
                           "begin_cluster_epoch" : None,
                           "stop_cluster_epoch" : None,
                           "top_k_list" : "[3, 5, 7]",
                           "test_batch_size" : 1024,
                           "eval_k" : 5,
                           "eval_metric" : "ndcg",
                           "random_seed" : random_seed,
                           "has_item_pool" : False,
                           "cluster_use_random_sort" : False,
                       })
wandb.run.name = f"invpref_{expt_num}"
os.makedirs(f"{save_dir}", exist_ok=True)

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
    
with open(test_data_path, 'r') as inp:
    inp.readline()
    lines: list = [line.rstrip() for line in inp.readlines()]
    print('Begin analyze raw test file')
    pairs, test_user_list, test_item_list = analyse_interaction_from_text(lines)
    ground_truth:list = analyes_user_interacted_set(pairs)
    inp.close()
    
if has_item_pool:
    item_pool_path: str = dataset_path + '/test_item_pool.csv'
    with open(item_pool_path, 'r') as inp:
        inp.readline()
        lines: list = [line.rstrip() for line in inp.readlines()]
        print('Begin analyze item pool file')
        pairs, _, _ = analyse_interaction_from_text(lines)
        
        item_pool: list = analyes_user_interacted_set(pairs)
        inp.close()
#item_pool은 사용자별 아이템 구매 리스트인데 빈공간이 없음

user_num = max(user_list + test_user_list) + 1
item_num = max(item_list + test_item_list) + 1

test_users_tensor: torch.LongTensor = torch.LongTensor(test_user_list)
test_users_tensor = test_users_tensor
sorted_ground_truth: list = [ground_truth[user_id] for user_id in test_user_list] # 사용자별 실제 구매상품

#%%
torch.manual_seed(random_seed)
np.random.seed(random_seed)

model = InvPrefImplicit(user_num = user_num,
                        item_num = item_num,
                        env_num = env_num,
                        factor_num = factor_num,
                        reg_only_embed = reg_only_embed,
                        reg_env_embed = reg_env_embed)
train_tensor: torch.LongTensor = torch.LongTensor(_train_data)


assert train_tensor.shape[1] == 3

envs_num: int = model.env_num
users_tensor: torch.Tensor = train_tensor[:,0]
items_tensor: torch.Tensor = train_tensor[:,1]
scores_tensor: torch.Tensor = train_tensor[:,2].float()
envs: torch.LongTensor = torch.LongTensor(np.random.randint(0, envs_num, _train_data.shape[0]))
optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr = lr)
recommend_loss_type = nn.BCELoss
cluster_distance_func = nn.BCELoss(reduction = 'none')
env_loss_type = nn.NLLLoss

batch_num = math.ceil(train_tensor.shape[0] / batch_size)

each_env_count = dict()
if alpha is None:
    alpha = 0.
    update_alpha = True
else:
    alpha = alpha
    update_alpha = False
    
sample_weights: torch.Tensor = torch.Tensor(np.zeros(train_tensor.shape[0]))
class_weights: torch.Tensor = torch.Tensor(np.zeros(envs_num))

eps_random_tensor: torch.Tensor = _init_eps(envs_num)

const_env_tensor_list: list = []

for env in range(envs_num):
    envs_tensor: torch.Tensor = torch.LongTensor(np.full(_train_data.shape[0], env, dtype = int))
    envs_tensor
    const_env_tensor_list.append(envs_tensor)
    
#%% Train start

cluster_diff_num_list: list = []
cluster_epoch_list: list = []
envs_cnt_list: list = []

loss_result_list: list = []
train_epoch_index_list: list = []

class_weights, sample_weights, result = stat_envs(envs, envs_num, scores_tensor)
# envs : 0,1 중 랜덤으로 뽑은 환경
# scores_tensor : train 데이터의 score
torch.save(model.state_dict(), f"{save_dir}/epoch_0.pt")

envs_tmp = deepcopy(envs)
envs_tmp = envs_tmp.numpy()
np.save(f"{save_dir}/env_epoch_0.npy", envs_tmp, allow_pickle=True)

for epoch_cnt in range(epochs):
    print(f"Epoch: {epoch_cnt}")
    
    model.train()
    loss_dicts_list: list = []
    for (batch_index, 
         (batch_users_tensor, 
          batch_items_tensor, 
          batch_scores_tensor,
          batch_envs_tensor,
          batch_sample_weights)) in tqdm(enumerate(mini_batch(batch_size, users_tensor, items_tensor, scores_tensor, envs, sample_weights))):
        if update_alpha:
            p = float(batch_index + (epoch_cnt + 1) * batch_num) / float((epoch_cnt + 1) * batch_num)
            alpha = 2./ (1. + np.exp(-10. * p)) - 1.
            
        invariant_score, env_aware_score, env_outputs = model(batch_users_tensor, batch_items_tensor, batch_envs_tensor, alpha)
            
        assert batch_users_tensor.shape == batch_items_tensor.shape == batch_scores_tensor.shape == batch_envs_tensor.shape
        assert batch_users_tensor.shape == invariant_score.shape
        assert invariant_score.shape == env_aware_score.shape
        assert env_outputs.shape[0] == env_aware_score.shape[0] and env_outputs.shape[1] == envs_num
            
        if use_class_re_weight:
            env_loss = env_loss_type(reduction = 'none')    
        else:
            env_loss = env_loss_type()
        if use_recommend_re_weight:
            recommed_loss = recommend_loss_type(reduction = 'none')
        else:
            recommend_loss =recommend_loss_type()
        
        invariant_loss: torch.Tensor = recommend_loss(invariant_score, batch_scores_tensor)
        env_aware_loss: torch.Tensor = recommend_loss(env_aware_score, batch_scores_tensor)
        
        envs_loss: torch.Tensor = env_loss(env_outputs, batch_envs_tensor)
        
        if use_class_re_weight:
            envs_loss = torch.mean(envs_loss * batch_sample_weights)
            
        if use_recommend_re_weight:
            invariant_loss = torch.mean(invariant_loss * batch_sample_weights)
            env_aware_loss = torch.mean(env_aware_loss * batch_sample_weights)
            
        L2_reg: torch.Tensor = model.get_L2_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)
        L1_reg: torch.Tensor = model.get_L1_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)
        
        loss: torch.Tensor = invariant_loss * invariant_coe + env_aware_loss * env_aware_coe + envs_loss * env_coe + L2_reg * L2_coe + L1_reg * L1_coe
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(invariant_loss * invariant_coe),
            'env_aware_loss': float(env_aware_loss * env_aware_coe),
            'envs_loss': float(envs_loss * env_coe),
            'L2_reg': float(L2_reg * L2_coe),
            'L1_reg': float(L1_reg * L1_coe),
            'loss': float(loss),
        }
        wandb_var.log(loss_dict)

    if (epoch_cnt % cluster_interval) == 0:

        if (begin_cluster_epoch is None or begin_cluster_epoch <= epoch_cnt) \
                and (stop_cluster_epoch is None or stop_cluster_epoch > epoch_cnt):
            print("clustering")
            envs, diff_num = cluster(
                deepcopy(model),
                envs,
                batch_size,
                users_tensor,
                items_tensor,
                scores_tensor,
                envs_num,
                const_env_tensor_list,
                cluster_distance_func,
                eps_random_tensor
                )

            wandb_var.log({"env_diff_num": diff_num})

        class_weights, sample_weights, result = stat_envs(envs, envs_num, scores_tensor)

    # if (epoch_cnt+1) % 1 == 0:
    if (epoch_cnt+1) in [10, 50, 100]:
        torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch_cnt+1}.pt")

        envs_tmp = deepcopy(envs)
        envs_tmp = envs_tmp.numpy()
        np.save(f"{save_dir}/env_epoch_{epoch_cnt+1}.npy", envs_tmp, allow_pickle=False)
#%%
