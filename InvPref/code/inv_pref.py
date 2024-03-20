#%%
import os
import math
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
#%%
def analys_interaction_from_text(lines: list, has_value: bool = False):
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
            pairs.append([user_id, item_id])
        