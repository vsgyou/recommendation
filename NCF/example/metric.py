import numpy as np
import torch
import torch.nn as nn

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    else:
        return 0.0
    
def metric(model, test_loader, top_k):
    HR, NDCG = [],[]
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