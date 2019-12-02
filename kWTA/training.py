import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable

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


def epoch(loader, model, opt=None, device=None, use_tqdm=False):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    if opt is None:
        model.eval()
    else:
        model.train()

    if use_tqdm:
        pbar = tqdm(total=len(loader))

    for X,y in loader:
        X,y = X.to(device), y.to(device)
  

        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_imagenet(loader, model, opt=None, device=None, use_tqdm=False):
    total_loss, total_err_top1, total_err_top5 = 0., 0., 0.

    if opt is None:
        model.eval()

    if use_tqdm:
        pbar = tqdm(total=len(loader))

    model.to(device)
    for X,y in loader:
        X,y = X.to(device), y.to(device)

        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err_top1 += (yp.max(dim=1)[1] != y).sum().item()

        _, pred = yp.topk(5, dim=1, sorted=True, largest=True)
        pred = pred.t()
        total_err_top5 += pred.eq(y.view(1,-1).expand_as(pred)).sum().item()

        total_loss += loss.item()*X.shape[0]

        if use_tqdm:
            pbar.update(1)

    return total_err_top1/len(loader.dataset), total_err_top5/len(loader.dataset), total_loss/len(loader.dataset)

def epoch_imagenet_adversarial(loader, model, device, attack, use_tqdm=False, n_test=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err_top1, total_err_top5 = 0., 0., 0.


    if use_tqdm:
        if n_test is None:
            pbar = tqdm(total=len(loader.dataset))
        else:
            pbar = tqdm(total=n_test)


    total_n = 0
    model.to(device)
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)

        total_err_top1 += (yp.max(dim=1)[1] != y).sum().item()
        _, pred = yp.topk(5, dim=1, sorted=True, largest=True)
        pred = pred.t()
        total_err_top5 += pred.eq(y.view(1,-1).expand_as(pred)).sum().item()
        total_loss += loss.item()*X.shape[0]

        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]

        if n_test is not None:
            if total_n >= n_test:
                break
        
    return total_err_top1/total_n, total_err_top5/total_n, total_loss/total_n


def epoch_func(loader, model, criterion, opt=None, device=None, use_tqdm=False):
    total_loss = 0.
    model.to(device)
    if use_tqdm:
        pbar = tqdm(total=len(loader))

    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = criterion(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_loss += loss.item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)

    return  total_loss / len(loader.dataset)
    
def epoch_distill_func(loader, model_teacher, model, device, opt=None, use_tqdm=True, n_test=None, loss_func='mse'):
    total_loss, total_err = 0.,0.
    total_n = 0

    model_teacher.to(device)
    model.to(device)

    if use_tqdm:
        if n_test is None:
            pbar = tqdm(total=len(loader.dataset))
        else:
            pbar = tqdm(total=n_test) 

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        teacher_output = model_teacher(X).detach()
        output = model(X)

        if loss_func=='mse':
            loss = nn.MSELoss()(output, teacher_output)
        elif loss_func=='l1':
            loss = nn.L1Loss()(output, teacher_output)
        elif loss_func=='kl':
            loss = nn.KLDivLoss()(F.log_softmax(output, dim=1), 
                                    F.softmax(teacher_output, dim=1))
        else:
            raise NotImplementedError

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += loss.item() * X.shape[0]
        total_n += X.shape[0]

        if use_tqdm:
            pbar.update(X.shape[0])
        
        if n_test is not None:
            if total_n > n_test:
                break
    
    return total_loss/total_n

def epoch_distill(loader, model_teacher, model, device, opt=None, use_tqdm=True, n_test=None, loss_func='mse'):
    total_loss, total_err = 0.,0.
    total_n = 0

    model_teacher.to(device)
    model.to(device)

    if use_tqdm:
        if n_test is None:
            pbar = tqdm(total=len(loader.dataset))
        else:
            pbar = tqdm(total=n_test) 

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        teacher_output = model_teacher(X).detach()
        output = model(X)

        if loss_func=='mse':
            loss = nn.MSELoss()(output, teacher_output)
        elif loss_func=='l1':
            loss = nn.L1Loss()(output, teacher_output)
        elif loss_func=='kl':
            loss = nn.KLDivLoss()(F.log_softmax(output, dim=1), 
                                    F.softmax(teacher_output, dim=1))
        else:
            raise NotImplementedError

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (output.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        total_n += X.shape[0]

        if use_tqdm:
            pbar.update(X.shape[0])
        
        if n_test is not None:
            if total_n > n_test:
                break
    
    return total_loss/total_n, total_err/total_n

def epoch_transfer_attack(loader, model_source, model_target, attack, device, success_only=False, use_tqdm=True, n_test=None, **kwargs):
    source_err = 0.
    target_err = 0.
    target_err2 = 0.

    success_total_n = 0
    

    model_source.eval()
    model_target.eval()

    total_n = 0

    if use_tqdm:
        pbar = tqdm(total=n_test)

    model_source.to(device)
    model_target.to(device)
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model_source, X, y, **kwargs)

        if success_only:
            raise NotImplementedError
        else:
            yp_target = model_target(X+delta).detach()
            yp_source = model_source(X+delta).detach()
            yp_origin = model_target(X).detach()
        source_err += (yp_source.max(dim=1)[1] != y).sum().item()
        target_err += (yp_target.max(dim=1)[1] != y).sum().item()
        target_err2 += (yp_origin.max(dim=1)[1] != y).sum().item()
        success_total_n += (yp_origin.max(dim=1)[1] == y)
        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]
        if n_test is not None:
            if total_n >= n_test:
                break

    return source_err / total_n, target_err / total_n, target_err2 /total_n


    # if randomize:
    #     delta = torch.rand_like(X, requires_grad=True)
    #     delta.data = delta.data * 2 * epsilon - epsilon
    # else:
    #     delta = torch.zeros_like(X, requires_grad=True)
        
    # for t in range(num_iter):
    #     loss = nn.CrossEntropyLoss()(model(X + delta), y)
    #     loss.backward()
    #     delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
    #     delta.grad.zero_()
    # return delta.detach()

def epoch_free_adversarial(loader, model, m, epsilon, opt, device, use_tqdm=False):
    """free adversarial training"""
    total_loss, total_err = 0.,0.
    total_n = 0

    pbar = tqdm(total=len(loader))


    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = torch.zeros_like(X, requires_grad=True)
        for i in range(m):
            model.train()
            yp = model(X+delta)
            loss_nn = nn.CrossEntropyLoss()(yp, y)

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss_nn.item() * X.shape[0]
            total_n += X.shape[0]

            #update network
            opt.zero_grad()
            loss_nn.backward()
            opt.step()

            #update perturbation
            delta.data = delta + epsilon*delta.grad.detach().sign()
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.grad.zero_()
        
        if use_tqdm:
            pbar.update(1)
    
    return total_err / total_n, total_loss / total_n


def epoch_ALP(loader, model, attack, alp_weight=0.5,
                opt=None, device=None, use_tqdm=False, n_test=None, **kwargs):
    """Adversarial Logit Pairing epoch over the dataset"""
    total_loss, total_err = 0.,0.

    # assert(opt is not None)
    model.train()

    if use_tqdm:
        if n_test is None:
            pbar = tqdm(total=len(loader.dataset))
        else:
            pbar = tqdm(total=n_test)
    total_n = 0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        model.eval()
        with torch.no_grad():
            clean_logit = model(X)
        delta = attack(model, X, y, **kwargs)
        
        model.train()
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y) + alp_weight*nn.MSELoss()(yp, clean_logit)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]

        if n_test is not None:
            if total_n >= n_test:
                break
        
    return total_err / total_n, total_loss / total_n

def epoch_adversarial(loader, model, attack, 
                opt=None, device=None, use_tqdm=False, n_test=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.

    if opt is None:
        model.eval()
    else:
        model.train()

    if use_tqdm:
        if n_test is None:
            pbar = tqdm(total=len(loader.dataset))
        else:
            pbar = tqdm(total=n_test)
    total_n = 0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        model.eval()
        delta = attack(model, X, y, **kwargs)
        if opt:
            model.train()
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if use_tqdm:
            pbar.update(X.shape[0])

        total_n += X.shape[0]

        if n_test is not None:
            if total_n >= n_test:
                break
        
    return total_err / total_n, total_loss / total_n

def get_activation(model, activation, name):
    def hook(model, input, output):
        activation[name] = output.cpu().detach()
    return hook

def register_layer(model, layer, activation, name):
    layer.register_forward_hook(get_activation(model, activation, name))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inp, target) in enumerate(val_loader):
        target = target.to(device)
        inp = inp.to(device)

        # compute output
        output = model(inp)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))
        top5.update(prec5.item(), inp.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def squared_l2_norm(x):
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            for idx_batch in range(batch_size):
                grad_idx = grad[idx_batch]
                grad_idx_norm = l2_norm(grad_idx)
                grad_idx /= (grad_idx_norm + 1e-8)
                x_adv[idx_batch] = x_adv[idx_batch].detach() + step_size * grad_idx
                eta_x_adv = x_adv[idx_batch] - x_natural[idx_batch]
                norm_eta = l2_norm(eta_x_adv)
                if norm_eta > epsilon:
                    eta_x_adv = eta_x_adv * epsilon / l2_norm(eta_x_adv)
                x_adv[idx_batch] = x_natural[idx_batch] + eta_x_adv
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def epoch_trade(loader, model, 
                opt, device=None, **kwargs):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        opt.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=opt,
                           **kwargs)
                        #    step_size=args.step_size,
                        #    epsilon=args.epsilon,
                        #    perturb_steps=args.num_steps,
                        #    beta=args.beta)
        loss.backward()
        opt.step()

    return 0, 0