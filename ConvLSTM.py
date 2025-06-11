import torch
import torch.nn as nn

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
