import torch
import torch.nn.functional as F
import torch.nn as nn

print('here new model')

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.drop = args.drop

        # self.fpredMLP = nn.Sequential(
        #     nn.Linear(306, 1000),
        #     nn.BatchNorm1d(1000),
        #     nn.ReLU(),
        #     nn.Dropout(p=args.drop),
        #     nn.Linear(1000, 306)
            
        # )
        self.extend = nn.Linear(51,256)
        self.dextend = nn.Linear(256, 51)
        self.fpredTF = torch.nn.TransformerEncoderLayer(d_model=256,nhead=4,
                                                  dim_feedforward=512,dropout=0.1,activation='relu')

        self.MLP3 = nn.Sequential(
            nn.Linear(306, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

            nn.Linear(1000, 360),

            nn.Sigmoid()

        )

    def forward(self, x):

        inputs = x[:,:306]

        # f_pred = self.fpredMLP(inputs)
        bs, L = inputs.shape
        inputs = inputs.reshape(bs, -1, 51)
        inputs = self.extend(inputs)

        f_pred = self.fpredTF(inputs.permute(1, 0, 2))
        f_pred = f_pred.permute(1,0,2)
        f_pred = self.dextend(f_pred).reshape(bs, -1)

        y_pred = self.MLP3(f_pred) # only the audio parts

        return y_pred, f_pred

