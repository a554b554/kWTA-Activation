import torch
import torch.nn as nn
import numpy as np
from kWTA import models
import copy

def register_layers(activation_list):
    for layer in activation_list:
        layer.record_activation()


def activation_counts(model, loader, activation_list, device, use_tqdm=True, test_size=None):
    count_list = []
    count = 0
    model.to(device)
    if use_tqdm:
        if test_size is not None:
            pbar = tqdm(total=test_size)
        else:
            pbar = tqdm(total=len(loader.dataset))

    for i, (X, y) in enumerate(loader):
        X = X.to(device)
        _ = model(X)
        for j, layer in enumerate(activation_list):
            act = layer.act
            batch_size = act.shape[0]
            if len(count_list) <= j:
                count_list.append(torch.zeros_like(act[0,:]))
            mask = (act!=0).to(act)
            mask_sum = mask.sum(dim=0)
            count_list[j] += mask_sum
        count += X.shape[0]
        if test_size is not None:
            if count >= test_size:
                break
        
        if use_tqdm:
            pbar.update(X.shape[0])
    return count_list

def append_activation_list(model, max_list_size):
    count = 0
    activation_list = []
    for (i,m) in enumerate(model.modules()):
        if isinstance(m, models.SparsifyBase):
            count += 1
            activation_list.append(m)
        if count>=max_list_size:
            break
    return activation_list


def get_mask_size(activation_list):
    size = 0
    for layer in activation_list:
        act = layer.act
        act = act.view(act.shape[0], -1)
        size += act.shape[1]
    return size


def compute_mask(model, X, activation_list):
    mask = None
    _ = model(X)
    for layer in activation_list:
        act = layer.act
        act = act.view(X.shape[0], -1)
        act_mask = act>0
        if mask is None:
            mask = act_mask
        else:
            mask = torch.cat((mask, act_mask), dim=1)
    return mask.to(dtype=torch.float32)

def compute_networkmask(model, loader, activation_list, device, max_n=None):
    model.to(device)
    all_label = None
    count = 0
    for i, (X,y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        if i == 0:
            _ = model(X)
            size = get_mask_size(activation_list)
            if max_n is not None:
                allmask = torch.zeros(max_n, size, dtype=torch.float32)
            else:
                allmask = torch.zeros(len(loader.dataset), size, dtype=torch.float32)
        current_mask = compute_mask(model, X, activation_list)
        allmask[i*X.shape[0]:(i+1)*X.shape[0], :].copy_(current_mask)


        current_sum = current_mask.sum().item()
        all_sum = allmask.sum().item()

        print('current mask:', current_sum, current_sum/current_mask.numel())
        print(i,'/',len(loader), all_sum , all_sum/allmask.numel())

        if all_label is None:
            all_label = y
        else:
            all_label = torch.cat((all_label, y))  

        count += X.shape[0]
        if max_n is not None:
            if count>= max_n:
                break
    
    return allmask, all_label.cpu()


def compute_networkmask_adv(model, loader, activation_list, device, attack, max_n=None, **kwargs):
    model.to(device)
    all_label = None
    count = 0
    for i, (X,y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        X = X+delta
        if i == 0:
            _ = model(X)
            size = get_mask_size(activation_list)
            if max_n is not None:
                allmask = torch.zeros(max_n, size, dtype=torch.float32)
            else:
                allmask = torch.zeros(len(loader.dataset), size, dtype=torch.float32)
        current_mask = compute_mask(model, X, activation_list, size)
        allmask[i*X.shape[0]:(i+1)*X.shape[0], :].copy_(current_mask)


        current_sum = current_mask.sum().item()
        all_sum = allmask.sum().item()

        print('current mask:', current_sum, current_sum/current_mask.numel())
        print(i,'/',len(loader), all_sum , all_sum/allmask.numel())

        if all_label is None:
            all_label = y
        else:
            all_label = torch.cat((all_label, y))  

        count += X.shape[0]
        if max_n is not None:
            if count>= max_n:
                break
    
    return allmask, all_label.cpu()

def kNN(model, labels, X, k, device, test_labels):
    model = model.to(device)
    X = X.to(device)
    correct = 0
    total = 0
    for i in range(X.shape[0]):
        x0 = X[i, :]
        sub = model-x0
        dist = torch.norm(sub, p=1, dim=1)
        mindist, idx = torch.topk(dist, k, largest=False)
        print('mindist', mindist.item(), 'predict label:', labels[idx].item(), 'true label:', test_labels[i].item())
        if labels[idx]==test_labels[i]:
            correct+=1
        total+=1
    return correct/total


def dist_stats1(loader, model, activation_list, class1, class2, n_test):

    dists = []
    for i, (X, y) in enumerate(loader):
        _ = model(X)
        print('batch', i, 'dists', len(dists))
        spl = int(X.shape[0]/2)
        mask = compute_mask(model, X, activation_list, get_mask_size(activation_list))
        for id1 in range(spl):
            if y[id1].item() != class1:
                continue
            for id2 in range(spl, spl*2):
                if y[id2].item() != class2:
                    continue
                dist = torch.norm(mask[id1,:]-mask[id2,:], p=1)
                dists.append(dist)
                if len(dists) >= n_test:
                    return dists
    return dists


def dist_stats2(loader, model, activation_list, class1, attack, n_test, **kwargs):

    dists = []
    for i, (X, y) in enumerate(loader):
        _ = model(X)
        print('batch', i, 'dists', len(dists))
        spl = int(X.shape[0])
        mask = compute_mask(model, X, activation_list, get_mask_size(activation_list))
        delta = attack(model, X, y, **kwargs)
        X_adv = X+delta
        _ = model(X_adv)
        mask_adv = compute_mask(model, X_adv, activation_list, get_mask_size(activation_list))
        for id1 in range(spl):
            if y[id1].item() != class1:
                continue
            dist = torch.norm(mask[id1,:]-mask_adv[id1,:], p=1)
            dists.append(dist)
            if len(dists) >= n_test:
                return dists
    return dists

def activation_pattern_cross(X, delta, step, batch_size, activation_list, model, device):
    cross_diff = []
    count= 0
    d_delta = delta/step
    assert(len(X.shape)==3)
    assert(step % batch_size == 0)
    model.to(device)
    while 1:
        T = torch.zeros(batch_size, X.shape[0], X.shape[1], X.shape[2])
        for i in range(batch_size):
            T[i,:,:,:] = X + count*d_delta
            count += 1 
        T = T.to(device)
        mask = compute_mask(model, T, activation_list)
        for i in range(mask.shape[0]-1):
            diff = torch.norm(mask[i+1,:]-mask[i,:], p=1)
            cross_diff.append(diff.item())
        
        
        if count >= step:
            break

    return cross_diff


def cross_diff_test(model, activation_list, X, y, 
        step, batch_size, eps, attack=None, **kwargs):
    if attack is not None:
        adv_delta = attack(model, X, y, epsilon=eps, **kwargs)
    
    device = next(model.parameters()).device

    stats0 = []
    stats5 = []
    stats10 = []

    for i in range(X.shape[0]):
        X0 = X[i,:,:,:]
        if attack is None:
            delta = torch.rand_like(X0)
            delta = delta.clamp(-eps, eps)
        else:
            delta = adv_delta[i,:,:,:].detach().cpu()
        cross_diff = activation_pattern_cross(X0, delta, device=device, step=step,
                    batch_size=batch_size, activation_list=activation_list, model=model)
        cross_diff = torch.FloatTensor(cross_diff)
        crossed = (cross_diff>0).sum().item()
        stats0.append(crossed)
        crossed = (cross_diff>5).sum().item()
        stats5.append(crossed)
        crossed = (cross_diff>10).sum().item()
        stats10.append(crossed)
    
    return torch.FloatTensor(stats0),torch.FloatTensor(stats5),torch.FloatTensor(stats10)