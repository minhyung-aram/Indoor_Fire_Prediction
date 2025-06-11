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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 하이퍼파라미터 설정
batch_size = 64
num_epochs = 100 



class ConvLSTMCell(nn.Module):
    """
    기본 ConvLSTM 셀 구현
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,  # 4배는 i, f, o, g 게이트를 위한 것
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        
        # 가중치 초기화
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        
        # 입력과 이전 은닉 상태 결합
        combined = torch.cat([x, h_prev], dim=1)
        
        # 컨볼루션 적용
        combined_conv = self.conv(combined)
        
        # 게이트 분리
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        # 게이트 활성화
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # 셀 상태와 은닉 상태 업데이트
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    여러 계층의 ConvLSTM 구현
    """
    def __init__(self, input_channels, hidden_channels_list, kernel_size, num_layers, batch_first=True):
        super(ConvLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels_list = hidden_channels_list
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # padding 계산 (same padding)
        self.padding = kernel_size // 2
        
        # ConvLSTM 셀 생성
        cell_list = []
        for i in range(self.num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels_list[i-1]
            cell_list.append(ConvLSTMCell(cur_input_channels,
                                         self.hidden_channels_list[i],
                                         self.kernel_size,
                                         self.padding))
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x, hidden_state=None):
        """
        Parameters
        ----------
        x : 입력 텐서, 크기는 (batch_size, seq_len, channels, height, width)
        hidden_state : 초기 은닉 상태, 기본값은 None (0으로 초기화)
        
        Returns
        -------
        layer_output_list : 각 레이어의 출력 리스트
        last_state_list : 각 레이어의 마지막 은닉 상태 리스트
        """
        # 배치 크기 및 시퀀스 길이 추출
        if self.batch_first:
            b, seq_len, c, h, w = x.size()
        else:
            seq_len, b, c, h, w = x.size()
            # (seq_len, batch, channel, height, width) -> (batch, seq_len, channel, height, width)
            x = x.permute(1, 0, 2, 3, 4)
        
        # 은닉 상태 초기화 (없는 경우)
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        # 각 레이어별 처리
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # 시퀀스의 각 시간 스텝 처리
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            # 현재 레이어 출력을 다음 레이어 입력으로 변환
            layer_output = torch.stack(output_inner, dim=1)
            x = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        # 반환 값 구성
        if not self.batch_first:
            layer_output_list = [layer_output.permute(1, 0, 2, 3, 4) for layer_output in layer_output_list]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class TemperatureConvLSTM(nn.Module):
    """
    화재 데이터를 위한 ConvLSTM 모델 구현
    입력: (batch_size, seq_length, features)
    출력: 온도, CO, 그을음 예측 이미지
    
    이 버전은 기존 TCN 모델처럼 Transposed CNN을 사용하여 출력 해상도를 높입니다.
    """
    def __init__(self, input_channels=1, hidden_channels_list=[16, 32, 64], kernel_size=3, num_layers=3):
        super(TemperatureConvLSTM, self).__init__()
        
        self.convlstm = ConvLSTM(
            input_channels=input_channels,
            hidden_channels_list=hidden_channels_list,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        
        # Reshape를 위한 FC 레이어
        self.fc = nn.Linear(7 * 10, 858)  # 원래 크기 7x10에서 26x33으로 변경
        
        # Upsampling 디코더
        self.transposed_conv = nn.ConvTranspose2d(
            64, 32, 
            kernel_size=3, 
            stride=2, 
            padding=(0, 0), 
            output_padding=(0, 1),
            bias=False
        )
        self.decoder_bn = nn.BatchNorm2d(32)
        self.decoder_relu = nn.ReLU()
        
        # 최종 출력 레이어 (온도, CO, 그을음 예측)
        self.output_conv = nn.Conv2d(
            in_channels=32, 
            out_channels=16,  # 3 채널: 온도, CO, 그을음
            kernel_size=3, 
            stride=1, 
            padding=(1, 0),
            bias=False
        )
        self.final_conv = nn.Conv2d(16, 8, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(16)
        self.final_relu = nn.ReLU()
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 입력 형태 확인 및 재구성
        batch_size, seq_length, n_features = x.shape
        
        # 데이터 재구성: (batch, seq, features) -> (batch, seq, channel, height, width)
        x_reshaped = x.reshape(batch_size, seq_length, 1, 7, 10)
        
        # ConvLSTM 통과
        layer_outputs, _ = self.convlstm(x_reshaped)
        
        # 마지막 레이어의 마지막 시간 스텝 출력 사용
        last_output = layer_outputs[-1][:, -1, :, :, :]  # (batch, channels, height, width)
        
        # FC 레이어를 이용한 공간 변환 준비
        # [batch, 32, 7, 10] -> [batch, 32, 7*10] -> [batch, 32, 26*33] -> [batch, 32, 26, 33]
        flattened = last_output.view(batch_size, 64, -1)  # [batch, 32, 7*10]
        reshaped = self.fc(flattened)  # [batch, 32, 26*33]
        decoded = reshaped.view(batch_size, 64, 26, 33)  # [batch, 32, 26, 33]
        
        # Transposed Convolution으로 해상도 증가
        upsampled = self.transposed_conv(decoded)  # [batch, 16, h_out*2, w_out*2]
        upsampled = self.decoder_bn(upsampled)
        upsampled = self.decoder_relu(upsampled)
        
        # 최종 컨볼루션으로 출력 생성
        output = self.output_conv(upsampled)  # [batch, 3, h_final, w_final]
        output = self.final_relu(self.final_bn(output))
        output = self.final_conv(output)
        # 개별 예측 맵 추출
        temp = output[:, 0, :, :]  # 온도 예측
        co = output[:, 1, :, :]    # CO 예측
        soot = output[:, 2, :, :]  # 그을음 예측
        
        return temp, co, soot

def save_checkpoint(model, epoch, loss_dict, checkpoint_path):
    """
    모델, 옵티마이저, 스케줄러 상태와 함께 손실 정보를 체크포인트로 저장합니다.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss_dict': loss_dict
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    체크포인트에서 모델, 옵티마이저, 스케줄러 상태를 로드합니다.
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
    
def train_model_2(model, train_dataloader, valid_dataloader, num_epochs=300, resume_path=None):
    """
    간결한 모델 학습 함수
    train_loss와 val_loss를 기록하고 그래프로 저장
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
            soot_loss = loss_func(soot_out, label[:, 2, :, :])
            
            # 총 손실
            total_loss = temp_loss + co_loss + soot_loss
            
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
            }, './ConvLSTM_checkpoints/conv_lstm.pth')

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
    
    print("학습 완료! 손실 그래프와 R² 그래프가 저장되었습니다.")

def evaluate_model(model, dataloader, device):
    """
    검증 또는 테스트 데이터로 모델을 평가합니다.
    """
    
    model.eval()
    total_loss = 0
    temp_loss = 0
    co_loss = 0
    soot_loss = 0
    
    loss_func = nn.MSELoss()
    
    # 예측값과 실제값을 누적할 배열들
    all_samples = 0
    
    # 각 변수별 예측/실제값 저장 배열
    all_temp_preds = []
    all_co_preds = []
    all_soot_preds = []
    all_temp_targets = []
    all_co_targets = []
    all_soot_targets = []
    
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device).float()
            batch_size = data.size(0)
            all_samples += batch_size
            
            # 모델 출력
            temp_out, co_out, soot_out = model(data)
            
            # 각 출력에 대한 손실 계산
            batch_temp_loss = loss_func(temp_out, label[:, 0, :, :])
            batch_co_loss = loss_func(co_out, label[:, 1, :, :])
            batch_soot_loss = loss_func(soot_out, label[:, 2, :, :])
            batch_total_loss = batch_temp_loss + batch_co_loss + batch_soot_loss
            
            # 손실 누적
            total_loss += batch_total_loss.item() * batch_size
            temp_loss += batch_temp_loss.item() * batch_size
            co_loss += batch_co_loss.item() * batch_size
            soot_loss += batch_soot_loss.item() * batch_size
            
            # 예측값과 실제값 수집 (메모리 효율성을 위해 numpy로 변환)
            all_temp_preds.append(temp_out.cpu().numpy())
            all_co_preds.append(co_out.cpu().numpy())
            all_soot_preds.append(soot_out.cpu().numpy())
            all_temp_targets.append(label[:, 0, :, :].cpu().numpy())
            all_co_targets.append(label[:, 1, :, :].cpu().numpy())
            all_soot_targets.append(label[:, 2, :, :].cpu().numpy())
    
    # 평균 손실 계산
    avg_loss = {
        'total': total_loss / all_samples,
        'temp': temp_loss / all_samples,
        'co': co_loss / all_samples,
        'soot': soot_loss / all_samples
    }
    
    # 예측값과 실제값 결합
    temp_preds = np.concatenate(all_temp_preds).flatten()
    co_preds = np.concatenate(all_co_preds).flatten()
    soot_preds = np.concatenate(all_soot_preds).flatten()
    
    temp_targets = np.concatenate(all_temp_targets).flatten()
    co_targets = np.concatenate(all_co_targets).flatten()
    soot_targets = np.concatenate(all_soot_targets).flatten()
    
    # R² 계산
    temp_r2 = r2_score(temp_targets, temp_preds)
    co_r2 = r2_score(co_targets, co_preds)
    soot_r2 = r2_score(soot_targets, soot_preds)
    
    metrics = {
        'loss': avg_loss,
        'r2': {
            'temp': temp_r2,
            'co': co_r2,
            'soot': soot_r2
        }
    }
    
    return metrics

def train_model(model, train_dataloader, valid_dataloader,num_epochs=1000, resume_path=None):
    """
    간결한 모델 학습 함수
    train_loss와 val_loss를 기록하고 그래프로 저장
    Early stopping 적용: 10 에포크 동안 0.0001 이상의 개선이 없으면 중단
    """
    
    # 최적화 설정 - 고정 학습률 0.0001 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
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
        start_epoch = checkpoint['epoch'] + 1
    
    # Early stopping 변수
    patience = 10  # 10 에포크 동안 개선이 없으면 중단
    threshold = 0.0001  # 이 값 이상의 개선만 개선으로 간주
    early_stop_counter = 0
        
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
            soot_loss = loss_func(soot_out, label[:, 2, :, :])
            
            # 총 손실
            total_loss = temp_loss + co_loss + soot_loss
            
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
        
        # Early stopping 로직
        if best_val_loss - avg_val_loss > threshold:  # threshold 이상 개선된 경우
            early_stop_counter = 0  # 카운터 리셋
            best_val_loss = avg_val_loss
            print(f"New best model with validation loss: {best_val_loss:.6f}")
            
            # 최적 모델 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, './ConvLSTM_checkpoints/test_model.pth')
        else:
            early_stop_counter += 1
            print(f"No significant improvement for {early_stop_counter} epochs. Best: {best_val_loss:.6f}, Current: {avg_val_loss:.6f}")
        
        # Early stopping 조건 확인
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {patience} epochs.")
            break
            
    # 훈련 종료 후 손실 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('./ConvLSTM_checkpoints/loss_plot.png')
    
    # R² 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(val_r2_temp, label='Temperature R²')
    plt.plot(val_r2_co, label='CO R²')
    plt.plot(val_r2_soot, label='Soot R²')
    plt.xlabel('Epochs')
    plt.ylabel('R² Score')
    plt.title('Validation R² Score')
    plt.legend()
    plt.savefig('./ConvLSTM_checkpoints/r2_plot.png')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r2_temp': val_r2_temp,
        'val_r2_co': val_r2_co,
        'val_r2_soot': val_r2_soot
    }

def test_model(model, test_dataloader, checkpoint_path=None):
    """
    테스트 데이터셋에서 모델을 평가하고 R² 점수를 반환합니다.
    """
    # 체크포인트에서 모델 로드 (제공된 경우)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path} for testing")
    
    print("Evaluating model on test data...")
    test_metrics = evaluate_model(model, test_dataloader, device)
    
    # 테스트 결과 출력
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {test_metrics['loss']['total']:.6f}")
    print(f"Temperature - Loss: {test_metrics['loss']['temp']:.6f}, R²: {test_metrics['r2']['temp']:.4f}")
    print(f"CO          - Loss: {test_metrics['loss']['co']:.6f}, R²: {test_metrics['r2']['co']:.4f}")
    print(f"Soot        - Loss: {test_metrics['loss']['soot']:.6f}, R²: {test_metrics['r2']['soot']:.4f}")
    print("="*50 + "\n")
    
    return test_metrics

def main():
    """
    메인 실행 함수
    """
    # 모델 초기화 및 디바이스 설정
    model = TemperatureConvLSTM().to(device)
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
    
    # test_dataloader = DataLoader(
    #     test_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=False
    # )
    
    print(f"Data loaders created with {num_workers} workers")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    # print(f"Test samples: {len(test_dataset)}")
    
    # 모델 학습
    print("Starting model training...")
    train_model_2(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        resume_path=None  # 학습 재개가 필요한 경우 체크포인트 경로 지정
    )
    
    # 최종 테스트 성능 평가
    # test_metrics = test_model(model, test_dataloader, checkpoint_path='./TCN_checkpoints/best_model.pth')

    
    # # 최종 결과 요약
    # print("\n" + "="*50)
    # print("FINAL TEST RESULTS SUMMARY")
    # print("="*50)
    # print("Best Overall Model R² scores:")
    # print(f"Temperature: {test_metrics['r2']['temp']:.4f}")
    # print(f"CO: {test_metrics['r2']['co']:.4f}")
    # print(f"Soot: {test_metrics['r2']['soot']:.4f}")
    # print("-"*50)
    # print("Best Individual Model R² scores:")
    # print("="*50 + "\n")
    
    # print("Training and evaluation complete!")

if __name__ == "__main__":
    main()