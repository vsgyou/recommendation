#%%
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
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