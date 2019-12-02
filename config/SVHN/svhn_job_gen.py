import json
import copy
import sys

import numpy as np


def regadv():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []

    model_list = ['ResNet18']
    batch_size = [(512,32)]



    for i, modelname in enumerate(model_list):
        job = {}
        job['eps'] = 0.047
        job['alpha'] = 0.01

        job['model'] = {}
        job['model']['name'] = modelname

        job['adv_train'] = {}
        job['adv_train']['attack'] = 'ml'

        job['logfilename'] = "./log/SVHN/{}_adv.log".format(modelname)
        job['savename'] = "./models/SVHN/{}_adv.pth".format(modelname)

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


def spnet1():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []

    model_list = ['spResNet18']

    batch_size = [(512,32)]

    # job['adv_train'] = {}
    # job['adv_train']['attack'] = 'ml'

    sp_list = [0.2, 0.1, 0.07, 0.03]

    for i, modelname in enumerate(model_list):
        for sp in sp_list:
            job = {}
            job['eps'] = 0.047
            job['alpha'] = 0.01

            job['model'] = {}
            job['model']['name'] = modelname
            job['model']['sp'] = sp

            job['logfilename'] = "./log/SVHN/{}_sp{}.log".format(modelname, sp)
            job['savename'] = "./models/SVHN/{}_sp{}.pth".format(modelname, sp)

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


def sp_adv():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []

    model_list = ['spResNet18']

    batch_size = [(512,32)]

    # job['adv_train'] = {}
    # job['adv_train']['attack'] = 'ml'

    sp_list = [0.1]

    for i, modelname in enumerate(model_list):
        for sp in sp_list:
            job = {}
            job['eps'] = 0.047
            job['alpha'] = 0.01

            job['model'] = {}
            job['model']['name'] = modelname
            job['model']['sp'] = sp

            job['adv_train'] = {}
            job['adv_train']['attack'] = 'ml'

            job['logfilename'] = "./log/SVHN/{}_sp{}_adv.log".format(modelname, sp)
            job['savename'] = "./models/SVHN/{}_sp{}_adv.pth".format(modelname, sp)

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
    # spnet1()
    # regadv()
    sp_adv()