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
#%%
def analyse_interaction_from_text(lines: list, has_value: bool = False):
    pairs : list = []
    users_set : set = set()
    items_set : set = set()
    
    for line in tqdm(lines):
        elements: list = line.split(',')
        user_id: int = int(elements[0])
        item_id: int = int(elements[1])
        if not has_value:
            pairs.append([user_id, item_id])
        else:
            value: float = float(elements[2])
            pairs.append([user_id, item_id, value])
        
        users_set.add(user_id)
        items_set.add(item_id)
    
    users_list: list = list(users_set)
    items_list: list = list(items_set)
    
    users_list.sort(reverse = False)
    items_list.sort(reverse = False)
    return pairs, users_list, items_list

def analyes_user_interacted_set(pairs: list):
    user_id_list: list = list()
    print('Int table...')
    for pair in tqdm(pairs):
        user_id, item_id = pair[0], pair[1]
        user_id_list.append(user_id)
    
    max_user_id: int = max(user_id_list)
    user_bought_map: list = [set() for i in range((max_user_id + 1))]
    print('Build mapping...')
    for pair in tqdm(pairs):
        user_id, item_id = pair[0], pair[1]
        user_bought_map[user_id].add(item_id)    
    return user_bought_map
# 유저 별 - 유저가 산 상품
def stat_envs(envs, envs_num, scores_tensor):
    result: dict = dict()
    class_rate_np: np.array = np.zeros(envs_num)
    for env in range(envs_num):
        cnt : int = int(torch.sum(envs == env))
        result[env] = cnt
        class_rate_np[env] = min(cnt+1, scores_tensor.shape[0] - 1)
        class_rate_np = class_rate_np / scores_tensor.shape[0]
        class_weights = torch.Tensor(class_rate_np)
        sample_weights = class_weights[envs]
    return class_weights, sample_weights, result

def mini_batch(batch_size: int, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i: i+batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i+batch_size] for x in tensors)
            
def cluster(model, envs, batch_size, users_tensor, items_tensor, scores_tensor, envs_num, const_env_tensor_list, cluster_distance_func, eps_random_tensor) -> int:
    cluster_use_random_sort = False
    model.eval()
    new_env_tensors_list: list = []
    
    for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) in enumerate(mini_batch(batch_size, users_tensor, items_tensor, scores_tensor)):
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
epochs = 1000
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
    
smaple_weights: torch.Tensor = torch.Tensor(np.zeros(train_tensor.shape[0]))
class_weights: torch.Tensor = torch.Tensor(np.zeros(envs_num))

eps_random_tensor: torch.Tensor = _init_eps(envs_num)

const_env_tensor_list: list = []

for env in range(envs_num):
    envs_tensor: torch.Tensor = torch.LongTensor(np.full(_train_data.shape[0], env, dtype = int))
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
          batch_sample_weights)) in tqdm(enumerate(mini_batch(batch_size, users_tensor, items_tensor, scores_tensor, envs, smaple_weights))):
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
        envs_tmp = envs_tmp.to("cpu").numpy()
        np.save(f"{save_dir}/env_epoch_{epoch_cnt+1}.npy", envs_tmp, allow_pickle=True)








#%%
class InvPrefExplicit(nn.Module):
    def __init__(self, 
                 user_num: int, 
                 item_num: int, 
                 env_num: int, 
                 factor_num: int, 
                 reg_only_embed: bool = False,
                 reg_env_embed: bool = True):
        super(InvPrefExplicit, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.env_num = env_num
        
        self.factor_num: int = factor_num
        
        self.embed_user_invariant = nn.Embedding(user_num, factor_num)
        self.embed_item_invariant = nn.Embedding(item_num, factor_num)
        
        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)
        
        self.embed_env = nn.Embedding(env_num, factor_num)
        
        self.env_classifier = LinearLogSoftMaxEnvClassifier(factor_num, env_num)
        
        self.reg_only_embed: bool = reg_only_embed
    
        self.reg_env_embed: bool = reg_env_embed
        
        self._init_weight()
        
    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std = 0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std = 0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std = 0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std = 0.01)
        nn.init.normal_(self.embed_env.weight, std = 0.01)
    
    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)
        
        users_embed_env_aware: torch.Tensor = self.embed_user_env_aware(users_id) 
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(items_id)
        
        envs_embed: torch.Tensor = self.embed_env(envs_id)
        
        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant # m_u,v
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed # a_u,v,e
        
        invariant_score: torch.Tensor = torch.sum(invariant_preferences, dim = 1) # x1_L
        env_aware_mid_score: torch.Tensor = torch.sum(env_aware_preferences, dim = 1) # x1_L
        env_aware_score: torch.Tensor = invariant_score + env_aware_mid_score # hat{y}_{u,v,e}
        
        reverse_invariant_preferences: torch.Tensor = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs: torch.Tensor = self.env_classifier(reverse_invariant_preferences)
        
        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)
        
    def get_items_reg(self, items_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_item_invariant(items_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) / (float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) / (float(len(items_id)) * float(self.facotr_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd!!')
        return reg_loss
        
    def get_env_reg(self, envs_id, norm: int):
        embed_gmf: torch.Tensor = self.embed_env(envs_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss
    
    def get_L2_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_rag()
            result = result + (self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 2)
        return result
    
    def get_L1_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 1)
        return result
    
    def predict(self, users_id, items_id):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)
        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        invariant_score: torch.Tensor = torch.sum(invariant_preferences, dim = 1)
        return invariant_score.reshape(-1)
    def cluster_predict(self, users_id, items_id, envs_id) -> torch.Tensor:
        _, env_aware_score, _ = self.forward(users_id, items_id, envs_id, 0.)
        return env_aware_score
# %%
class LinearLogSoftMaxEnvClassifier(nn.Module):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim = 1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)
        
    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result
    
    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num + torch.norm(self.linear_map.bias, 1) / self.bias_num
    
    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)
#%%        
result = linear_map(reverse_invariant_preferences)
result = classifier_func(result)
     
#%%
        
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
def _init_eps(envs_num):
    base_eps = 1e-10
    eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(envs_num)]
    temp: torch.Tensor = torch.Tensor(eps_list)
    eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))
    
    return eps_random_tensor

class InvPrefImplicit(nn.Module):
    def __init__(self, 
                user_num: int, 
                item_num: int, 
                env_num: int, 
                factor_num: int, 
                reg_only_embed: bool = False,
                reg_env_embed: bool = True):
        super(InvPrefImplicit, self).__init__()
        self.user_num: int = user_num
        self.item_num: int = item_num
        self.env_num: int = env_num
        
        self.factor_num: int = factor_num
        
        self.embed_user_invariant = nn.Embedding(user_num, factor_num)
        self.embed_item_invariant = nn.Embedding(item_num, factor_num)
        
        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)
        
        self.embed_env = nn.Embedding(env_num, factor_num)
        
        self.env_classifier = LinearLogSoftMaxEnvClassifier(factor_num, env_num)
        self.output_func = nn.Sigmoid()
        
        self.reg_only_embed: bool = reg_only_embed
        self.reg_env_embed: bool = reg_env_embed
        
        self._init_weight()
        
    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_env.weight, std=0.01)
    
    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)
        
        users_embed_env_aware: torch.Tensor = self.embed_user_env_aware(users_id)
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(items_id)
        
        envs_embed: torch.Tensor = self.embed_env(envs_id)
        
        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed
        
        invariant_score : torch.Tensor = self.output_func(torch.sum(invariant_preferences, dim = 1))
        env_aware_mid_score: torch.Tensor = self.output_func(torch.sum(env_aware_preferences, dim = 1))
        env_aware_score: torch.Tensor = invariant_score * env_aware_mid_score
        
        reverse_invariant_preferences: torch.Tensor = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs: torch.Tensor = self.env_classifier(reverse_invariant_preferences)
        
        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)
    
    def get_users_reg(self, users_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_user_invariant(users_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_user_env_aware(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_item_invariant(items_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_envs_reg(self, envs_id, norm: int):
        embed_gmf: torch.Tensor = self.embed_env(envs_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L2_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_reg()
            result = result + (
                    self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 2)
        # print('L2', result)
        return result

    def get_L1_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (
                    self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 1)
        # print('L2', result)
        return result

    def predict(self, users_id):
        users_embed_gmf: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_gmf: torch.Tensor = self.embed_item_invariant.weight

        user_to_cat = []
        for i in range(users_embed_gmf.shape[0]):
            tmp: torch.Tensor = users_embed_gmf[i:i + 1, :]
            tmp = tmp.repeat(items_embed_gmf.shape[0], 1)
            user_to_cat.append(tmp)
        users_emb_cat: torch.Tensor = torch.cat(user_to_cat, dim=0)
        items_emb_cat: torch.Tensor = items_embed_gmf.repeat(users_embed_gmf.shape[0], 1)

        invariant_preferences: torch.Tensor = users_emb_cat * items_emb_cat
        invariant_score: torch.Tensor = self.output_func(torch.sum(invariant_preferences, dim=1))
        return invariant_score.reshape(users_id.shape[0], items_embed_gmf.shape[0])

    def cluster_predict(self, users_id, items_id, envs_id) -> torch.Tensor:
        _, env_aware_score, _ = self.forward(users_id, items_id, envs_id, 0.)
        return env_aware_score










# %%
