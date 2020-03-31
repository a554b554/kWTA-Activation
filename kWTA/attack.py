import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import torchvision

def gen_rand_labels(y, num_classes):
    targets = torch.randint_like(y, low=0, high=num_classes)
    for i in range(len(targets)):
        while targets[i]==y[i]:
            targets[i] = torch.randint(low=0, high=10, size=(1,))
    return targets


def gen_least_likely_labels(model, X):
    preds = model(X)
    return preds.min(dim=1)[1]

def fgsm_linf_untargeted(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


def pgd_l2_untargeted(model, X, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
        
    return delta.detach()

def pgd_linf_untargeted(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
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

def pgd_linf_untargeted2(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        yp = model(X+delta)
        loss = - yp.gather(1,y[:,None])[:,0]
        loss = loss.sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_untargeted_mostlikely(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    yp = model(X)
    y = yp.max(dim=1)[1]
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf_untargeted_maxce(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    yp = model(X).detach()

    for t in range(num_iter):
        loss = nn.KLDivLoss()(F.log_softmax(model(X+delta), dim=1), F.softmax(yp, dim=1))
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf_targ(model, X, y, epsilon=0.1, alpha=0.01, use_y=True,
         num_iter=20, y_targ='rand', num_classes=10, randomize=False):
    """ Construct targeted adversarial examples on the examples X"""

    if isinstance(y_targ, str):
        strlist = ['rand', 'leastlikely']
        assert(y_targ in strlist)
        if y_targ == 'rand':
            y_targ = gen_rand_labels(y, num_classes)
        elif y_targ == 'leastlikely':
            y_targ = gen_least_likely_labels(model, X)


    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        loss = yp[:,y_targ]
        if not use_y:
            y = yp.max(dim=1)[1]
        loss = loss - yp.gather(1,y[:,None])[:,0]
        loss = loss.sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_targ3(model, X, y, epsilon=0.1, alpha=0.01,
         num_iter=20, y_targ='rand', num_classes=10, randomize=False):
    """ Without using the label information"""

    if isinstance(y_targ, str):
        strlist = ['rand', 'leastlikely']
        assert(y_targ in strlist)
        if y_targ == 'rand':
            y_targ = gen_rand_labels(y, num_classes)
        elif y_targ == 'leastlikely':
            y_targ = gen_least_likely_labels(model, X)


    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        loss = yp[:,y_targ].sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_targ2(model, X, y, epsilon=0.1, alpha=0.01, 
                    num_iter=20, y_targ='rand', num_classes=10, randomize=False):
    """ Construct targeted adversarial examples on the examples X"""
    
    if isinstance(y_targ, str):
        strlist = ['rand', 'leastlikely']
        assert(y_targ in strlist)
        if y_targ == 'rand':
            y_targ = gen_rand_labels(y, num_classes)
        elif y_targ == 'leastlikely':
            y_targ = gen_least_likely_labels(model, X)


    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)


    for t in range(num_iter):
        yp = model(X + delta)
        loss = 2*yp[:,y_targ].sum() - yp.sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_targ4(model, X, y, epsilon=0.1, alpha=0.01, 
                    num_iter=20, y_targ='rand', num_classes=10, randomize=False):
    """ Construct targeted adversarial examples on the examples X"""
    
    if isinstance(y_targ, str):
        strlist = ['rand', 'leastlikely']
        assert(y_targ in strlist)
        if y_targ == 'rand':
            y_targ = gen_rand_labels(y, num_classes)
        elif y_targ == 'leastlikely':
            y_targ = gen_least_likely_labels(model, X)


    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)


    for t in range(num_iter):
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss(yp, y) - nn.CrossEntropyLoss(yp, y_targ)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def deepfool(model, X, y, epsilon=0.1, num_iter=50):
    model.eval()
    delta = torch.zeros_like(X)
    X = X.clone()
    X.requires_grad_()

    out = model(X+delta)
    n_class = out.shape[1]
    py = out.max(1)[1].item()
    ny = out.max(1)[1].item()

    i_iter = 0

    while py == ny and i_iter < num_iter:
        out[0, py].backward(retain_graph=True)
        grad_np = X.grad.data.clone()
        value_l = np.inf
        ri = None

        for i in range(n_class):
            if i == py:
                continue

            X.grad.data.zero_()
            out[0, i].backward(retain_graph=True)
            grad_i = X.grad.data.clone()

            wi = grad_i - grad_np
            fi = out[0, i] - out[0, py]
            value_i = np.abs(fi.item()) / np.linalg.norm(wi.cpu().numpy().flatten())

            if value_i < value_l:
                ri = value_i/np.linalg.norm(wi.cpu().numpy().flatten()) * wi

        delta += ri.clone()
        X.grad.data.zero_()
        out = model(X+delta)
        py = out.max(1)[1].item()
        i_iter += 1
    
    delta = delta.clamp(-epsilon, epsilon)
    
    return delta.detach()



def one_pixel_perturb(p, img):
    img_size = img.size(1)

def one_pixel_evolve():
    pass

def one_pixel_attack(model, X, y):
    raise NotImplementedError
