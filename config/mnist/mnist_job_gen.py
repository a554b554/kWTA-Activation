import json
import copy
import sys

import numpy as np


def spDNN():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []
    job1 = {}


    # sp_list = [0.3, 0.2, 0.1, 0.05, 0.01]
    sp_list = [0.008, 0.005, 0.003, 0.002, 0.001]

    # hs = np.linspace(1000, 10000, 5)
    hs = [10000]
    for h in hs:
        for sp in sp_list:
            job = {}

            job['eps'] = 0.3
            job['alpha'] = 0.01

            job['model'] = {}
            job['model']['name'] = 'spDNN'
            job['model']['hidden_size'] = int(h)
            job['model']['sp'] = sp
            job['logfilename'] = "./log/MNIST/{}_sp{}_h{}.log".format(job['model']['name'],
                job['model']['sp'], job['model']['hidden_size'])
            job['savename'] = "./models/MNIST/{}_sp{}_h{}.pth".format(job['model']['name'],
                job['model']['sp'], job['model']['hidden_size'])
            job['epoch'] = 20
            job['adv_train'] = False
            job['lr'] = 1e-2
            job['momentum'] = 0.9

            job['train_batch_size'] = 200
            job['test_batch_size'] = 100
            job['n_test_adv'] = 1000
            config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)


def DNN():
    filename = sys.argv[1]

    config = {}
    config['jobs'] = []
    job1 = {}


    sp_list = [0.3, 0.2, 0.1, 0.05, 0.01]

    hs = np.linspace(1000, 10000, 5)
    for h in hs:
        job = {}

        job['eps'] = 0.3
        job['alpha'] = 0.01

        job['model'] = {}
        job['model']['name'] = 'DNN'
        job['model']['hidden_size'] = int(h)
        job['logfilename'] = "./log/MNIST/{}_h{}.log".format(job['model']['name'],
             job['model']['hidden_size'])
        job['savename'] = "./models/MNIST/{}_h{}.pth".format(job['model']['name'],
             job['model']['hidden_size'])
        job['epoch'] = 20
        job['adv_train'] = False
        job['lr'] = 1e-2
        job['momentum'] = 0.9

        job['train_batch_size'] = 200
        job['test_batch_size'] = 100
        job['n_test_adv'] = 1000
        config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)


def CNN():
    filename = sys.argv[1]
    config = {}
    config['jobs'] = []
    job1 = {}
    channels = np.array([32,32,64,64])
    factor_list = [1,2,4]
    for factor in factor_list:
        job = {}

        job['eps'] = 0.3
        job['alpha'] = 0.01

        job['model'] = {}
        job['model']['name'] = 'CNN'
        job['model']['hidden_size'] = 20000
        c = channels*factor
        job['model']['channels'] = [int(c[0]), int(c[1]), int(c[2]), int(c[3])]


        job['logfilename'] = "./log/MNIST/{}_f{}.log".format(job['model']['name'],
             factor)
        job['savename'] = "./models/MNIST/{}_f{}.pth".format(job['model']['name'],
             factor)
        job['epoch'] = 20
        job['adv_train'] = False
        job['lr'] = 1e-2
        job['momentum'] = 0.9

        job['train_batch_size'] = 200
        job['test_batch_size'] = 100
        job['n_test_adv'] = 1000
        config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)

def spCNN():
    filename = sys.argv[1]
    config = {}
    config['jobs'] = []
    job1 = {}
    sp_list = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.007, 0.005]
    # sp_list = [0.007, 0.004, 0.001]
    channels = np.array([32,32,64,64])
    factor_list = [1,2,4]
    for factor in factor_list:
        for sp in sp_list:
            job = {}

            job['eps'] = 0.3
            job['alpha'] = 0.01

            job['model'] = {}
            job['model']['name'] = 'spCNN'
            job['model']['hidden_size'] = 20000
            c = channels*factor
            job['model']['channels'] = [int(c[0]), int(c[1]), int(c[2]), int(c[3])]
            job['model']['sp1'] = sp
            job['model']['sp2'] = sp

            job['logfilename'] = "./log/MNIST/{}_f{}_sp{}.log".format(job['model']['name'],
                factor, sp)
            job['savename'] = "./models/MNIST/{}_f{}_sp{}.pth".format(job['model']['name'],
                factor, sp)
            job['epoch'] = 20
            job['adv_train'] = False
            job['lr'] = 1e-2
            job['momentum'] = 0.9

            job['train_batch_size'] = 200
            job['test_batch_size'] = 100
            job['n_test_adv'] = 1000
            config['jobs'].append(job)

    j = json.dumps(config, indent=4)
    with open(filename, "w+") as f:
        f.write(j)

if __name__ == "__main__":
    # spDNN()
    # DNN()
    spCNN()
    # CNN()