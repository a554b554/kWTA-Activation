import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision
from kWTA import models
import copy

from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import axes3d, Axes3D

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def showimg(img, cmap=None):
    if not isinstance(img, np.ndarray):
        npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest', cmap=cmap)

def showallimg(imgs, n_row, normalize=True, cmap=None):
    fig=plt.figure(figsize=(30, 30), dpi=20, facecolor='w', edgecolor='k')
    showimg(torchvision.utils.make_grid(imgs.to("cpu").detach(), padding=1, normalize=normalize, nrow=n_row), cmap=cmap)

def show_batch_img(batch, img_id, n_row, normalize=True, figsize=30, dpi=20):
    fig=plt.figure(figsize=(figsize, figsize), dpi=dpi, facecolor='w', edgecolor='k')
    imgs = batch[img_id:img_id+1,:,:,:].permute(1,0,2,3)
    showimg(torchvision.utils.make_grid(imgs.to("cpu").detach(), padding=1, normalize=normalize, nrow=n_row))

def compare_imgs(imgs1, imgs2, n_row, normalize=True, figsize=30, dpi=20):
    toshow = torch.zeros(imgs1.shape[0]*2, imgs1.shape[1], imgs1.shape[2], imgs1.shape[3])
    for i in range(imgs1.shape[0]):
        toshow[2*i,:,:,:] = imgs1[i,:,:,:]
        toshow[2*i+1,:,:,:] = imgs2[i,:,:,:]
    fig=plt.figure(figsize=(figsize, figsize), dpi=dpi, facecolor='w', edgecolor='k')
    showimg(torchvision.utils.make_grid(toshow.to("cpu").detach(), padding=1, normalize=normalize, nrow=n_row))


def draw_loss(model, X, y, epsilon, device, dir1='grad', grid_size=25, batch_size=100):
    Xi, Yi = np.meshgrid(np.linspace(-epsilon, epsilon, grid_size), np.linspace(-epsilon,epsilon,grid_size))
    print(Xi.shape)
    def grad_at_delta(delta):
        delta.requires_grad_(True)
        nn.CrossEntropyLoss()(model(X+delta), y[0:1]).backward()
        return delta.grad.detach().sign().view(-1).cpu().numpy()

    assert(dir1 in ['grad', 'rand'])

    if dir1 == 'grad':
        dir1 = grad_at_delta(torch.zeros_like(X, requires_grad=True))
    elif dir1 == 'rand':
        dir1 = np.sign(np.random.randn(dir1.shape[0]))
    delta2 = torch.zeros_like(X, requires_grad=True)
    delta2.data = torch.tensor(dir1).view_as(X).to(device)
    # dir2 = grad_at_delta(delta2)
    np.random.seed(0)
    dir2 = np.sign(np.random.randn(dir1.shape[0]))
    
    all_deltas = torch.tensor((np.array([Xi.flatten(), Yi.flatten()]).T @ 
                              np.array([dir2, dir1])).astype(np.float32))

    all_deltas = all_deltas.view(-1,X.shape[1],X.shape[2],X.shape[3])
    start = 0
    yp = torch.zeros(all_deltas.shape[0], 10)
    while 1:
        end = start+batch_size
        if end>all_deltas.shape[0]:
            end=all_deltas.shape[0]
        deltas = all_deltas[start:end,:,:,:].to(device)
        yp0 = model(deltas + X).detach().cpu()
        yp[start:end, :] = yp0
        start = end
        if start >= all_deltas.shape[0]:
            break
    Zi = nn.CrossEntropyLoss(reduction="none")(yp, y.cpu()[0:1].repeat(yp.shape[0])).detach().cpu().numpy()
    print(yp.shape)
    Zi = Zi.reshape(*Xi.shape)
    #Zi = (Zi-Zi.min())/(Zi.max() - Zi.min())
    return Xi, Yi, Zi
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')
    # ls = LightSource(azdeg=0, altdeg=200)
    # rgb = ls.shade(Zi, plt.cm.coolwarm)

    # surf = ax.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, linewidth=0,
    #                    antialiased=True, facecolors=rgb)

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def count_feature_stats(activations, layer_list, imgs=None):
    for l in layer_list:
        print(l)
        act = activations[l]
        if imgs is not None:
            act = act[imgs,:,:,:]
        g = torch.norm(act, p=0).item()
        l = torch.sum(act<=0).item()
        print('total:', act.shape)
        print('>0:', g)
        print('<=0:', l)
        print('ratio:', g/(g+l))

def feature_diff(act1, act2, layer_list, relu=False):
    diff = {}
    for l in layer_list:
        diff[l] = act1[l] - act2[l]
        if relu:
            diff[l] = nn.ReLU()(act1[l]) - nn.ReLU()(act2[l])
    return diff


def feature_diff_stats(act1, act2, relu=False):
    diff = act1 - act2
    print('act1 l0:', torch.norm(act1, p=0).item(), 'act2 l0:', torch.norm(act2, p=0).item(), 'diff l2:', torch.norm(diff, p=2).item(), 'diff l0:', torch.norm(diff, p=0).item())
    # print('act2 l0:', torch.norm(act1, p=0))
    # print('diff l2:', torch.norm(diff, p=2))
    # print('diff l0:', torch.norm(diff, p=0))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# def feature_counts(fea)

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


def perturb_network(model, eps, whitelist=None, blacklist=None):

    assert((not whitelist) or (not blacklist))

    for m in model.modules():
        if whitelist:
            if m not in whitelist:
                continue
        elif blacklist:
            if m in blacklist:
                continue
        
        if hasattr(m, 'weight') and (m.weight is not None):
            delta = eps*torch.rand_like(m.weight.data)
            m.weight.data += delta

        if hasattr(m, 'bias') and (m.bias is not None):
            delta = eps*torch.rand_like(m.bias.data)
            m.bias.data += delta

    return model


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


def visualize_prediction_trajectory(model, loader, eps, step, device, vid=0, attack=None, **kwargs):
    X, y = next(iter(loader))
    X, y = X.to(device), y.to(device)
    model.to(device)
    if attack is None:
        delta = torch.rand_like(X)
    else:
        delta = attack(model, X, y, **kwargs)
    delta = delta.clamp(-eps, eps)
    delta /= step
    vals = []
    for i in range(step):
        T = X+i*delta
        yp = model(T)
        vals.append(yp[vid,:].cpu().detach().numpy())
    vals = np.array(vals)
    plt.figure()
    for i in range(10):
        plt.plot(np.linspace(0, eps, step), vals[:,i], '-', markersize=1)

def gen_rand_labels(y, num_classes):
    targets = torch.randint_like(y, low=0, high=num_classes)
    for i in range(len(targets)):
        while targets[i]==y[i]:
            targets[i] = torch.randint(low=0, high=10, size=(1,))
    return targets

def adversarial_network(model, X, y, lr, step, device, attack='untarg'):
    if attack == 'targ':
        y_targ = gen_rand_labels(y, num_classes=10)
    
    adv_model = copy.deepcopy(model)
    predictions = []
    losses = []
    errs = []
    opt = optim.SGD(adv_model.parameters(), lr=lr, momentum=0.9)

    adv_model.to(device)
    X, y = X.to(device), y.to(device)
    for i in range(step):
        yp = adv_model(X)
        y0 = yp.max(dim=1)[1]
        c_errs = (yp.max(dim=1)[1] != y).sum().item()/X.shape[0]
        errs.append(c_errs)
        opt.zero_grad()

        if attack == 'untarg':
            loss = -nn.CrossEntropyLoss()(yp, y)
        elif attack == 'targ':
            loss = -yp[:, y_targ] + yp.gather(1,y[:,None])[:,0]
        else:
            raise NotImplementedError

        loss.backward()
        opt.step()
        losses.append(loss.item())
        predictions.append(yp.detach().cpu().numpy())
    predictions = np.array(predictions)
    return predictions, losses, errs