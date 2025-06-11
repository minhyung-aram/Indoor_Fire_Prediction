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
from torch.nn.utils.parametrizations import weight_norm
import torch.optim as optim
import math
from sklearn.metrics import r2_score
from tqdm import tqdm

# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 하이퍼파라미터 설정
batch_size = 64 
dropout_tcn = 0.3  # TCN 블록 내 dropout
num_epochs = 300    # 에폭 수 50으로 제한
width = 10
height = 7
# 체크포인트 저장 디렉토리 생성
# os.makedirs('./checkpoints', exist_ok=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        # 첫번째 conv
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 3, 3),
            padding=(((kernel_size-1) * dilation) // 2, 1, 1),
            dilation=(dilation, 1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_tcn)
        
        # 두번째 conv
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 3, 3),
            padding=(((kernel_size-1) * dilation) // 2, 1, 1), # 왼쪽에만 패딩
            dilation=(dilation, 1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_tcn)
        
        # skip connection에서 채널을 맞춰주기 위한 conv
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
        self.final_activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        # channel이 맞지 않는다면 1x1x1 conv사용
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # gradient flow의 안정화를 위해서 final activation으로 relu 사용
        return self.final_activation(out + residual)

class TemperatureTCN(nn.Module):
    def __init__(self, seq_length=30, input_features=70, num_channels=[8, 16, 32, 64], kernel_size=3):
        super(TemperatureTCN, self).__init__()
        
        self.last_channel = num_channels[-1]
        # TCN 블록 구성
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # ResidualBlock 추가
            layers.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        self.total_fc = nn.Linear(70, 858)
        self.transposecnn = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=(0, 0), output_padding=(0,1), bias=False)
        self.encode_cnn = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=(1, 0), bias=False)
        self.decode_cnn = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        self.relu = nn.ReLU()
        self.final_relu = nn.ReLU()
        self.final_bn = nn.BatchNorm2d(16)
        self.bn = nn.BatchNorm2d(32)
    
    def forward(self, x):
        # 입력 형태 확인 및 재구성
        batch_size, seq_length, n_features = x.shape
        
        # TCN 처리를 위한 5D 형태로 변환 (3D conv는 batch 차원 빼고 4차원 입력을 받음)
        x = x.view(batch_size, 1, seq_length, height, width)
        
        # TCN 블록 통과
        tcn_out = self.tcn(x)  # [batch, channels, seq, 7, 10]
        # TCN Layer의 마지막 시퀀스 표현만 이용, 이후 FC 통과를 위해 reshape
        out = tcn_out[:,:,-1,:,:].reshape(batch_size, self.last_channel, height * width)
        # Channel-wise FC 통과 (레퍼 논문은 Sequence-wise FC)
        fc_out = self.total_fc(out)
        # upsampling을 위해 reshape
        fc_out = fc_out.reshape(batch_size, self.last_channel, 26, 33) # 이건 샘플이 달라질때마다 조정
        # upsampling (batch, channel, 53, 68)
        up_out = self.transposecnn(fc_out) 
        up_out = self.bn(up_out)
        up_out = self.relu(up_out)
        # 크기 조정 cnn (batch, chanel, 53, 66)
        encode_out = self.encode_cnn(up_out)
        encode_out = self.final_bn(encode_out)
        encode_out = self.final_relu(encode_out)
        # 마지막 특징 추출 1x1 cnn (batch, chanel, 53, 66)
        final_out = self.decode_cnn(encode_out)
        # 3가지 정보 추출
        temp = final_out[:,0,:,:]
        co = final_out[:,1,:,:]
        soot = final_out[:,2,:,:]
        
        return tuple([temp, co, soot])
    

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    체크포인트에서 모델, 옵티마이저, 스케줄러 상태를 로드
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss_dict = checkpoint.get('loss_dict', {})
        
        print(f"Checkpoint loaded successfully. Resuming from epoch {epoch}")
        return epoch, loss_dict
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, {}


def train_model(model, train_dataloader, valid_dataloader, num_epochs=300, resume_path=None):
    """
    모델 학습 함수1, 스케줄러를 사용하여 지속적으로 lr을 줄임
    """
    
    # 최적화 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # 간단한 학습 기록
    train_losses = []
    val_losses = []
    val_r2_temp = []
    val_r2_co = []
    val_r2_soot = []
    
    # 손실 함수
    loss_func = nn.MSELoss()
    
    # 체크포인트 로드 (재개 학습)
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
    # 학습 루프
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0
        
        # 학습 단계
        for batch_idx, (data, label) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # 데이터를 디바이스로 이동
            data, label = data.to(device), label.to(device).float()
            
            # 모델 출력
            temp_out, co_out, soot_out = model(data)
            
            # 각 출력에 대한 손실 계산
            temp_loss = loss_func(temp_out, label[:, 0, :, :])
            co_loss = loss_func(co_out, label[:, 1, :, :])
            soot_loss = loss_func(soot_out, label[:, 2, :, :],)
            
            # 총 손실
            total_loss =  temp_loss + co_loss + soot_loss
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 손실 누적
            batch_size = data.size(0)
            epoch_train_loss += total_loss.item() * batch_size
        
        # 에폭당 평균 학습 손실 계산
        avg_train_loss = epoch_train_loss / len(train_dataloader.dataset)
        
        # 검증 단계
        model.eval()
        val_loss = 0
        
        # R² 계산을 위한 변수들
        all_temp_preds = []
        all_co_preds = []
        all_soot_preds = []
        all_temp_true = []
        all_co_true = []
        all_soot_true = []
        
        with torch.no_grad():
            for data, label in valid_dataloader:
                data, label = data.to(device), label.to(device).float()
                
                # 모델 출력
                temp_out, co_out, soot_out = model(data)
                
                # 각 출력에 대한 손실 계산
                temp_loss = loss_func(temp_out, label[:, 0, :, :])
                co_loss = loss_func(co_out, label[:, 1, :, :])
                soot_loss = loss_func(soot_out, label[:, 2, :, :])
                
                # 총 손실
                total_loss = temp_loss + co_loss + soot_loss
                
                # 손실 누적
                batch_size = data.size(0)
                val_loss += total_loss.item() * batch_size
                
                # R² 계산을 위해 예측값과 실제값 수집
                all_temp_preds.append(temp_out.cpu().numpy())
                all_co_preds.append(co_out.cpu().numpy())
                all_soot_preds.append(soot_out.cpu().numpy())
                all_temp_true.append(label[:, 0, :, :].cpu().numpy())
                all_co_true.append(label[:, 1, :, :].cpu().numpy())
                all_soot_true.append(label[:, 2, :, :].cpu().numpy())
        
        
        all_temp_preds = np.concatenate(all_temp_preds, axis=0).flatten()
        all_co_preds = np.concatenate(all_co_preds, axis=0).flatten()
        all_soot_preds = np.concatenate(all_soot_preds, axis=0).flatten()
        all_temp_true = np.concatenate(all_temp_true, axis=0).flatten()
        all_co_true = np.concatenate(all_co_true, axis=0).flatten()
        all_soot_true = np.concatenate(all_soot_true, axis=0).flatten()
        
        # R² 점수 계산
        temp_r2 = r2_score(all_temp_true, all_temp_preds)
        co_r2 = r2_score(all_co_true, all_co_preds)
        soot_r2 = r2_score(all_soot_true, all_soot_preds)
        
        # 에폭당 평균 검증 손실 계산
        avg_val_loss = val_loss / len(valid_dataloader.dataset)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 학습 기록 업데이트
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_r2_temp.append(temp_r2)
        val_r2_co.append(co_r2)
        val_r2_soot.append(soot_r2)
        
        # 학습 진행 상황 출력
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_val_loss:.6f}")
        print(f"Valid R²: Temp: {temp_r2:.4f}, CO: {co_r2:.4f}, Soot: {soot_r2:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        # 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model with validation loss: {best_val_loss:.6f}")
            torch.save({
                'model_state_dict': model.state_dict(),
            }, './TCN_final_checkpoints/TCN_only.pth')

        if optimizer.param_groups[0]['lr'] < 0.00001:
            break
    
    # 손실 그래프 저장
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('./TCN_loss_graph.png')
    # plt.close()
    
    # # R² 그래프 저장
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(val_r2_temp) + 1), val_r2_temp, label='Temperature R²')
    # plt.plot(range(1, len(val_r2_co) + 1), val_r2_co, label='CO R²')
    # plt.plot(range(1, len(val_r2_soot) + 1), val_r2_soot, label='Soot R²')
    # plt.xlabel('Epochs')
    # plt.ylabel('R² Score')
    # plt.title('Validation R² Scores')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('./TCN_r2_graph.png')
    # plt.close()
    
    # print("학습 완료! 손실 그래프와 R² 그래프가 저장되었습니다.")
    
    return train_losses, val_losses, val_r2_temp, val_r2_co, val_r2_soot


def main():
    """
    메인 실행 함수
    """
    # 모델 초기화 및 디바이스 설정
    model = TemperatureTCN().to(device)
    print(f"Model initialized on {device}")
    
    # 데이터 로딩
    dataset = myDataset("dataset")
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # 데이터셋 분할
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = int(dataset_size * 0.2)
    # valid_size = int(dataset_size * 0.2)
    # test_size = dataset_size - train_size - valid_size
    
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size])
    
    # DataLoader 설정 - num_workers는 시스템에 맞게 조정
    num_workers = min(16, os.cpu_count() or 1)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 검증 데이터는 섞지 않음
        num_workers=num_workers, 
        pin_memory=True
    )
    
    print(f"Data loaders created with {num_workers} workers")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # 모델 학습
    print("Starting model training...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        resume_path=None  # 학습 재개가 필요한 경우 체크포인트 경로 지정
    )

if __name__ == "__main__":
    main()