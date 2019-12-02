import json
import copy
import sys

import numpy as np


def regularnet():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []

    model_list = ['DenseNet121', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
    batch_size = [(64,32), (128,32), (128,32), (64,32), (32,16), (32,16)]

    for i, modelname in enumerate(model_list):
        job = {}
        job['eps'] = 0.031
        job['alpha'] = 0.007

        job['model'] = {}
        job['model']['name'] = modelname

        job['logfilename'] = "./log/CIFAR/{}.log".format(modelname)
        job['savename'] = "./models/CIFAR/{}.pth".format(modelname)

        job['epoch'] = 100
        job['epoch1'] = 60
        job['epoch2'] = 80

        job['lr'] = 0.1
        job['momentum'] = 0.9
        job['epoch1_lr'] = 0.01
        job['epoch2_lr'] = 0.001

        job['test_attack'] = ['untarg1', 'untarg2']
        job['n_test_adv'] = 1000
        job['train_batch_size'] = batch_size[i][0]
        job['test_batch_size'] = batch_size[i][1]
        config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)

def spwideresnet():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = [] 

    depth = 22
    width_list = [6, 8, 10, 12]
    sp_list = [0.2, 0.1, 0.05]

    for width in width_list:
        for sp in sp_list:
            job = {}
            job['eps'] = 0.031
            job['alpha'] = 0.007

            job['model'] = {}
            job['model']['name'] = 'spWideResNet'
            job['model']['depth'] = depth
            job['model']['width'] = width
            job['model']['sp'] = sp

            job['logfilename'] = "./log/CIFAR/{}_d{}_w{}_sp{}.log".format(job['model']['name'], depth, width, sp)
            job['savename'] = "./models/CIFAR/{}_d{}_w{}_sp{}.pth".format(job['model']['name'], depth, width, sp)

            job['epoch'] = 100
            job['epoch1'] = 60
            job['epoch2'] = 80

            job['lr'] = 0.1
            job['momentum'] = 0.9
            job['epoch1_lr'] = 0.01
            job['epoch2_lr'] = 0.001

            job['test_attack'] = ['untarg1', 'untarg2']
            job['n_test_adv'] = 1000
            job['train_batch_size'] = 64
            job['test_batch_size'] = 32
            config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)


def sp_adv():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []

    # model_list = ['spDenseNet121', 'spResNet18', 'spResNet101']
    # model_list = ['spResNet34', 'spResNet50', 'spResNet152']


    # batch_size = [(128,32), (512,32), (128,32)]
    # batch_size = [(512,32), (256,32), (128,32)]


    sp = 0.1
    # modelname = 'spResNet18'
    modelname = 'spWideResNet'




    job = {}
    job['eps'] = 0.031
    job['alpha'] = 0.007

    job['model'] = {}
    job['model']['name'] = modelname
    job['model']['sp'] = sp

    job['model']['depth'] = 22
    job['model']['width'] = 10

    job['adv_train'] = {}
    job['adv_train']['attack'] = 'ml'

    job['logfilename'] = "./log/CIFAR/{}_sp{}_adv.log".format(modelname, sp)
    job['savename'] = "./models/CIFAR/{}_sp{}_adv.pth".format(modelname, sp)

    job['epoch'] = 100
    job['epoch1'] = 60
    job['epoch2'] = 80

    job['lr'] = 0.1
    job['momentum'] = 0.9
    job['epoch1_lr'] = 0.01
    job['epoch2_lr'] = 0.001

    job['test_attack'] = ['untarg1', 'untarg2']
    job['n_test_adv'] = 1000
    job['train_batch_size'] = 512
    job['test_batch_size'] = 32
    config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)

def reg_adv():
    # model_list = ['ResNet18', 'ResNet152', 'DenseNet121']
    # batch_size = [(512,32), (64,32), (128,32)]
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []

    # model_list = ['ResNet18']
    # batch_size = [(512,32)]

    # model_list = ['ResNet152']
    # batch_size = [(64,32)]

    model_list = ['DenseNet121']
    batch_size = [(128,32)]

    for i, modelname in enumerate(model_list):
        job = {}
        job['eps'] = 0.031
        job['alpha'] = 0.007

        job['model'] = {}
        job['model']['name'] = modelname

        job['adv_train'] = {}
        job['adv_train']['attack'] = 'ml'

        job['logfilename'] = "./log/CIFAR/{}_adv.log".format(modelname)
        job['savename'] = "./models/CIFAR/{}_adv.pth".format(modelname)

        job['epoch'] = 100
        job['epoch1'] = 60
        job['epoch2'] = 80

        job['lr'] = 0.1
        job['momentum'] = 0.9
        job['epoch1_lr'] = 0.01
        job['epoch2_lr'] = 0.001

        job['test_attack'] = ['untarg1', 'untarg2']
        job['n_test_adv'] = 1000
        job['train_batch_size'] = batch_size[i][0]
        job['test_batch_size'] = batch_size[i][1]
        config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)


def spwidereg():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = [] 

    depth = 22
    width_list = [6, 8, 10, 12]

    for width in width_list:
        job = {}
        job['eps'] = 0.031
        job['alpha'] = 0.007

        job['model'] = {}
        job['model']['name'] = 'WideResNet'
        job['model']['depth'] = depth
        job['model']['width'] = width

        job['logfilename'] = "./log/CIFAR/{}_d{}_w{}.log".format(job['model']['name'], depth, width)
        job['savename'] = "./models/CIFAR/{}_d{}_w{}.pth".format(job['model']['name'], depth, width)

        job['epoch'] = 100
        job['epoch1'] = 60
        job['epoch2'] = 80

        job['lr'] = 0.1
        job['momentum'] = 0.9
        job['epoch1_lr'] = 0.01
        job['epoch2_lr'] = 0.001

        job['test_attack'] = ['untarg1', 'untarg2']
        job['n_test_adv'] = 1000
        job['train_batch_size'] = 64
        job['test_batch_size'] = 32
        config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)

def spnet1():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []

    # model_list = ['spDenseNet121', 'spResNet18', 'spResNet101']
    model_list = ['spResNet34', 'spResNet50', 'spResNet152']

    # batch_size = [(128,32), (512,32), (128,32)]
    batch_size = [(512,32), (256,32), (128,32)]


    sp_list = [0.3, 0.2, 0.1, 0.07, 0.03, 0.01]

    for i, modelname in enumerate(model_list):
        for sp in sp_list:
            job = {}
            job['eps'] = 0.031
            job['alpha'] = 0.007

            job['model'] = {}
            job['model']['name'] = modelname
            job['model']['sp'] = sp

            job['logfilename'] = "./log/CIFAR/{}_sp{}.log".format(modelname, sp)
            job['savename'] = "./models/CIFAR/{}_sp{}.pth".format(modelname, sp)

            job['epoch'] = 100
            job['epoch1'] = 60
            job['epoch2'] = 80

            job['lr'] = 0.1
            job['momentum'] = 0.9
            job['epoch1_lr'] = 0.01
            job['epoch2_lr'] = 0.001

            job['test_attack'] = ['untarg1', 'untarg2']
            job['n_test_adv'] = 1000
            job['train_batch_size'] = batch_size[i][0]
            job['test_batch_size'] = batch_size[i][1]
            config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)


if __name__ == "__main__":
    # regularnet()
    # spnet1()
    # reg_adv()
    # spwideresnet()
    # sp_adv()
    spwidereg()