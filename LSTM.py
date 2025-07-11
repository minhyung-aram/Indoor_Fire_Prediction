import torch
import torch.nn as nn

class TemperatureLSTM(nn.Module):
    '''
    레퍼 논문의 LSTM 모델을 그대로 구현한 모델입니다.
    입력: (batch_size, seq_length, features)
    출력: 온도, CO, 그을음 예측 이미지
    '''
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
        fc_out = self.fc(lstm_out)         # Sequence-wise FC, (batch, 30, 858)
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
