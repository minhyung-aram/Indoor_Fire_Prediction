import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 5D inputs
    """
    def __init__(self, channels, reduction_ratio=8):
        super(SEBlock, self).__init__()
        
        # 채널 수가 reduction_ratio보다 작은 경우 조정
        reduced_channels = max(channels // reduction_ratio, 1)
        
        # 3D 데이터에 대한 전역 평균 풀링
        self.squeeze = nn.AdaptiveAvgPool3d(1)  # 시간과 공간 차원 모두 압축
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, seq_len, height, width = x.size()
        
        # 시간 및 공간 차원 압축 (Squeeze)
        y = self.squeeze(x).view(batch_size, channels)
        
        # 채널 간 관계 모델링 (Excitation)
        y = self.excitation(y).view(batch_size, channels, 1, 1, 1)
        
        # 스케일링 적용
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate=0.2):
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
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 두번째 conv
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 3, 3),
            padding=(((kernel_size-1) * dilation) // 2, 1, 1),
            dilation=(dilation, 1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
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

        self.se = SEBlock(out_channels)
        self.se2 = SEBlock(out_channels)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.se(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.se2(out)
        
        # channel이 맞지 않는다면 1x1x1 conv사용
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # gradient flow의 안정화를 위해서 final activation으로 relu 사용
        return self.final_activation(out + residual)


class TemperatureTCN(nn.Module):
    def __init__(self, seq_length=30, input_features=70, num_channels=[8, 16, 32, 64], 
                 kernel_size=3, dropout_rate=0.2, height=7, width=10, 
                 fc_output_features=858, upsample_size=(26, 33)):
        super(TemperatureTCN, self).__init__()
        
        self.last_channel = num_channels[-1]
        self.height = height
        self.width = width
        self.fc_output_features = fc_output_features
        self.upsample_size = upsample_size
        
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
                    dilation=dilation,
                    dropout_rate=dropout_rate
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        self.total_fc = nn.Linear(height * width, fc_output_features)
        self.transposecnn = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=(0, 0), 
            output_padding=(0, 1), bias=False
        )
        self.encode_cnn = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, 
            stride=1, padding=(1, 0), bias=False
        )
        self.decode_cnn = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        self.relu = nn.ReLU()
        self.final_relu = nn.ReLU()
        self.final_bn = nn.BatchNorm2d(16)
        self.bn = nn.BatchNorm2d(32)
    
    def forward(self, x):
        # 입력 형태 확인 및 재구성
        batch_size, seq_length, n_features = x.shape
        
        # TCN 처리를 위한 5D 형태로 변환
        x = x.view(batch_size, 1, seq_length, self.height, self.width)
        
        # TCN 블록 통과
        tcn_out = self.tcn(x)  # [batch, channels, seq, height, width]
        
        # TCN Layer의 마지막 시퀀스 표현만 이용, 이후 FC 통과를 위해 reshape
        out = tcn_out[:, :, -1, :, :].reshape(batch_size, self.last_channel, self.height * self.width)
        
        # Channel-wise FC 통과
        fc_out = self.total_fc(out)
        
        # upsampling을 위해 reshape
        fc_out = fc_out.reshape(batch_size, self.last_channel, *self.upsample_size)
        
        # upsampling
        up_out = self.transposecnn(fc_out)
        up_out = self.bn(up_out)
        up_out = self.relu(up_out)
        
        # 크기 조정 cnn
        encode_out = self.encode_cnn(up_out)
        encode_out = self.final_bn(encode_out)
        encode_out = self.final_relu(encode_out)
        
        # 마지막 특징 추출 1x1 cnn
        final_out = self.decode_cnn(encode_out)
        
        # 3가지 정보 추출
        temp = final_out[:, 0, :, :]
        co = final_out[:, 1, :, :]
        soot = final_out[:, 2, :, :]
        
        return temp, co, soot
    