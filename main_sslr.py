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

sys.path.append('modelclass')
sys.path.append('funcs')
torch.manual_seed(7)  # For reproducibility across different computers
torch.cuda.manual_seed(7)

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))       # 打印按指定格式排版的时间


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xinyuan experiments')
    parser.add_argument('-gpuidx', metavar='gpuidx', type=int, default=0, help='gpu number')
    parser.add_argument('-epoch', metavar='EPOCH', type=int, default=10)
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


    args = parser.parse_args()

# savemodel=False
BATCH_SIZE = args.batch
print(sys.argv[1:])
print("experiments - xinyuan")

device = torch.device("cuda:{}".format(args.gpuidx) if torch.cuda.is_available() else 'cpu')
args.device = device
print(device)


def training(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, 0):

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

def testing(Xte, Yte, Ite):  # Xte: feature, Yte: binary flag
    model.eval()
    print('start testing')
    Y_pred_t=[]
    for ist in range(0, len(Xte), BATCH_SIZE):
        ied = np.min([ist+BATCH_SIZE, len(Xte)])
        inputs = Variable(torch.from_numpy(Xte[ist:ied])).type(torch.FloatTensor).to(device)
        output = model.forward(inputs)
        Y_pred_t.extend(output.cpu().detach().numpy()) # in CPU

    # ------------ error evaluate   ----------
    MAE1, ACC1, MAE2, ACC2,_,_,_,_ = funcs.MAEeval(Y_pred_t, Yte, Ite)
    torch.cuda.empty_cache()
    return MAE1, MAE2, ACC1, ACC2

# ############################# load the data and the model ##############################################################
modelname = args.model  
lossname='MSE'

models, criterion = loaddata.Dataextract(modelname, lossname)

train_loader, Xtr, Ytr, Itr, Ztr, Xte1, Yte1, Ite1, Xte2, Yte2, Ite2, GT1, GT2, GTtr = dataread.dataread(BATCH_SIZE, args) # <--- logger to be added
train_loader_obj = funcs.MyDataloaderClass(Xtr, Ztr)  # Xtr-data feature, Ztr-Gaussian-format label
train_loader = DataLoader(dataset=train_loader_obj, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

model = models.Model(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)
print(model)

######## Training + Testing #######
EP = args.epoch
MAEh1, MAEh2, ACCh1, ACCh2, MAEl1, MAEl2, ACCl1, ACCl2 = np.zeros(EP), np.zeros(EP), np.zeros(EP), np.zeros(
    EP), np.zeros(EP), np.zeros(EP), np.zeros(EP), np.zeros(EP)
model_dict = {}
# max_loss = 100
plt.figure
for ep in range(EP):
    training(ep)
    MAEl1[ep], MAEl2[ep], ACCl1[ep], ACCl2[ep] = testing(Xte2, Yte2, Ite2)  # loudspeaker
    MAEh1[ep], MAEh2[ep], ACCh1[ep], ACCh2[ep] = testing(Xte1, Yte1, Ite1)  # human - real face detection
    # # --------- display the result -----------
    mae, acc, _ = funcs.display(args.model, ep, EP, MAEl1, MAEl2, ACCl1, ACCl2, MAEh1, MAEh2, ACCh1, ACCh2, Ite1, Ite2)


print("finish all!")

