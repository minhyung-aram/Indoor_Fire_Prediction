import torch
import torch.nn as nn
from processing_data2nd import *
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import glob
import matplotlib.pylab as plt
import random
from tqdm import tqdm
import torch.optim as optim
import math
from sklearn.metrics import r2_score

# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 하이퍼파라미터 설정
batch_size = 64
dropout_fc = 0.1   # FC 레이어 dropout
num_epochs = 300  # 에폭 수 30으로 제한

# 체크포인트 저장 디렉토리 생성
os.makedirs('./lstm_checkpoints', exist_ok=True)

class TemperatureLSTM(nn.Module):
    def __init__(self, input_size=70, num_layers=2):
        super(TemperatureLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, 256, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256, 858)
        self.transposecnn = nn.ConvTranspose2d(30, 16, kernel_size=3, stride=2, padding=(0, 0), output_padding=(0,1))
        self.cnn = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=(1, 0))
        self.final_cnn = nn.Conv2d(8, 3, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(8)
        self.final_relu = nn.ReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv_bn = nn.BatchNorm2d(16)  # 채널 수를 명시해야 함
    
    def forward(self, x):
        # 배치 크기 동적 처리
        batch_size = x.size(0)
        
        lstm_out, (hn, cn) = self.lstm(x)  # (batch, 30, 512)
        fc_out = self.fc(lstm_out)         # (batch, 30, 858)
        fc_out = fc_out.reshape(batch_size, 30, 26, 33)
        transposecnn_out = self.transposecnn(fc_out)  # torch.Size([batch, 16, 53, 68])
        transposecnn_out = self.conv_bn(transposecnn_out)
        transposecnn_out = self.relu(transposecnn_out)
        cnn_out = self.cnn(transposecnn_out)  # torch.Size([batch, 3, 53, 66])
        out = self.final_relu(self.final_bn(cnn_out))
        cnn_out = self.final_cnn(out)
        # 각 출력 채널 분리 (temp, co, soot)
        temp_out = cnn_out[:, 0, :, :]  # (batch, 53, 66)
        co_out = cnn_out[:, 1, :, :]    # (batch, 53, 66)
        soot_out = cnn_out[:, 2, :, :]  # (batch, 53, 66)
        
        return temp_out, co_out, soot_out
