import numpy as np
import torch
from torch.autograd import Variable
import hdf5storage
import argparse
from torch.utils.data import DataLoader
import sys
import dataread
import time
import os
import random
import loaddata
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale as zscale # Z-score normalizatin: mean-0, std-1
import funcs
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
# import allnoise
import torch.nn as nn


sys.path.append('modelclass')
sys.path.append('funcs')
torch.manual_seed(7)  # For reproducibility across different computers
torch.cuda.manual_seed(7)

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))       # 打印按指定格式排版的时间


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xinyuan experiments')
    parser.add_argument('-gpuidx', metavar='gpuidx', type=int, default=0, help='gpu number')
    parser.add_argument('-epoch', metavar='EPOCH', type=int, default=30)
    parser.add_argument('-drop', metavar='drop', type=float, default=0.2)
    parser.add_argument('-lr', metavar='lr', type=float, default=0.001)
    parser.add_argument('-trA', metavar='trA', type=str, default='gcc') # Options:  
    parser.add_argument('-teA', metavar='teA', type=str, default='gcc') # Options: gcc, melgcc
    parser.add_argument('-trV', metavar='trV', type=str, default=None) # Options: 'faceSSLR'
    parser.add_argument('-teV', metavar='teV', type=str, default=None) #  
    parser.add_argument('-model', metavar='model', type=str, default='MLP3') # Options: modelMLP3attentsoftmax
    parser.add_argument('-batch', metavar='batch', type=int, default=2**8)
    parser.add_argument('-train', metavar='train', type=int, default=1)  # whether need training set
    # parser.add_argument('-test', metavar='eval', type=int, default=1)  # whether the evaluation mode
    parser.add_argument('-Vy', metavar='Vy', type=int, default=1)  # whether use the vertical video feature
    parser.add_argument('-VO', metavar='VO', type=int, default=0)  # train and test on frames with face
    parser.add_argument('-datapath', type=str, default='/mntcephfs/lee_dataset/loc/ICASSP2021data')  # train and test on frames with face
    parser.add_argument('-upbound', type=int, default=0)  # whether is the upperbound
    parser.add_argument('-Hidden', default=5000, type=int,help='')
    parser.add_argument('-phaseN', type=int, default=10)  # total incremetnal learning step number
    parser.add_argument('-incremental', type=int, default=0)  # whether is the upperbound
    parser.add_argument('-recurbase', type=int, default=0)  # whether is the upperbound


    args = parser.parse_args()

# savemodel=False
BATCH_SIZE = args.batch
print(sys.argv[1:])
print("experiments - xinyuan")

device = torch.device("cuda:{}".format(args.gpuidx) if torch.cuda.is_available() else 'cpu')
args.device = device
print(device)


def training(epoch, Xtr, Ztr, Itr, GTtr, phase, args):
    model.train()
    phaseN= args.phaseN

    GTstep=360/phaseN
    ICLrange=[0,(phase+1)*GTstep] if args.upbound else [phase*GTstep,(phase+1)*GTstep]

    Xtr, Ztr, Itr, GTtr, DoArange=funcs.ICLselect(Xtr, Ztr, Itr, GTtr, ICLrange, 'Train')
    train_loader_obj = funcs.MyDataloaderClass(Xtr, Ztr, Itr, GTtr)  # Xtr-data feature, Ztr-Gaussian-format label
    train_loader = DataLoader(dataset=train_loader_obj, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,drop_last=True)

    for batch_idx, (data, target, num, gtlabel) in enumerate(train_loader, 0):
        # data: input feature
        # target: 360-Gaussian distribution label

        inputs, target = Variable(data).type(torch.FloatTensor).to(device), Variable(target).type(torch.FloatTensor).to(device)

        # start training
        y_pred = model.forward(inputs)  # return the predicted angle
        loss = criterion(y_pred.double(), target.double())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if (round(train_loader.__len__()/5/100)*100)>0 and batch_idx % (round(train_loader.__len__()/5/100)*100) == 0:
            print("training - epoch%d-batch%d: loss=%.3f" % (epoch, batch_idx, loss.data.item()))

    torch.cuda.empty_cache()

def testing(ep, Xte, Yte, Ite, GT, phase, phaseN):  # Xte: feature, Yte: binary flag
    model.eval()

    GTstep=360/phaseN
    ICLrange=[0,(phase+1)*GTstep]

    Xte, Yte, Ite, GT, DoArange=funcs.ICLselect(Xte, Yte, Ite, GT, ICLrange, 'Test')

    print('start testing')
    Y_pred_t=[]
    for ist in range(0, len(Xte), BATCH_SIZE):
        ied = np.min([ist+BATCH_SIZE, len(Xte)])
        inputs = Variable(torch.from_numpy(Xte[ist:ied])).type(torch.FloatTensor).to(device)
        output = model.forward(inputs)
        Y_pred_t.extend(output.cpu().detach().numpy()) # in CPU

    # ------------ error evaluate   ----------
    MAE1, ACC1, MAE2, ACC2,_,_,_,_ = funcs.MAEeval(Y_pred_t, Yte, Ite)
    print(DoArange+" ep=%1d phase=%1d Testing MAE1: %.1f MAE2: %.1f | ACC1: %.1f ACC2: %.1f " % (ep, phase, MAE1, MAE2, ACC1, ACC2))

    torch.cuda.empty_cache()
    return MAE1, MAE2, ACC1, ACC2

# ############################# load the data and the model ##############################################################
Xtr, Ytr, Itr, Ztr, Xte1, Yte1, Ite1, Xte2, Yte2, Ite2, GT1, GT2, GTtr = dataread.dataread(BATCH_SIZE, args) # <--- logger to be added
modelname = args.model  
lossname='MSE'

models, criterion = loaddata.Dataextract(modelname, lossname)
model = models.Model(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)
print(model)
# h,b,p=plt.hist(GTtr,bins=360)


######## Training + Testing #######
EP = args.epoch
MAEl1, MAEl2, ACCl1, ACCl2 = np.zeros(args.phaseN), np.zeros(args.phaseN), np.zeros(args.phaseN), np.zeros(args.phaseN)
model_dict = {}

# incremental learning
forget_rate1 = np.zeros(args.phaseN)
forget_rate2 = np.zeros(args.phaseN)


for ep in range(EP):
    phase=ep//(EP//args.phaseN)
    print('Incremental learning  epoch '+str(ep)+'  phase '+str(phase))

    training(ep, Xtr, Ztr, Itr, GTtr, phase, args)

    if ep%(EP//args.phaseN)==args.epoch/args.phaseN-1:
        MAEl1[phase], MAEl2[phase], ACCl1[phase], ACCl2[phase] = testing(ep, Xte2, Yte2, Ite2, GT2, phase, args.phaseN)  # loudspeaker
        forget_rate1[phase] = ACCl1[0]-ACCl1[phase]
        forget_rate2[phase] = ACCl2[0]-ACCl2[phase]
    # MAEh1[ep], MAEh2[ep], ACCh1[ep], ACCh2[ep] = testing(ep, Xte1, Yte1, Ite1, GT1, phase, args.phaseN)  # human - real face detection
    # # --------- display the result -----------
    # mae, acc, _ = funcs.display(args.model, ep, EP, MAEl1, MAEl2, ACCl1, ACCl2, MAEh1, MAEh2, ACCh1, ACCh2, Ite1, Ite2)
    
        print("Forgeting rate for Phase %01d/%01d: ACC1 %.2f ACC2 %.2f" % (phase, args.phaseN, forget_rate1[phase], forget_rate2[phase]))

print("finish all! average testing MAE1: %.1f MAE2: %.1f | ACC1: %.1f ACC2: %.1f " % (np.mean(MAEl1), np.mean(MAEl2), np.mean(ACCl1), np.mean(ACCl2)))

