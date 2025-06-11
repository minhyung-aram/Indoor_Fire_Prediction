import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import glob
import joblib

class myDataset(Dataset):
    def __init__(self, dataset_dir, seq_length=30, future_step=60, time_step=1):
        """
        테스트용 데이터 클래스입니다. 반환되는 데이터의 seq_length를 자유롭게 조절할 수 있도록 설계한거 빼곤 
        processing_data.py와 같습니다. 
        
        Args:
            dataset_dir (str): 데이터셋 디렉토리 경로
            seq_length (int): 입력 시퀀스 길이 (몇 개의 연속된 시간 포인트를 사용할지)
            future_step (int): 예측할 미래 시점의 스텝 수 (현재 시점으로부터 몇 스텝 이후를 예측할지)
            time_step (float): 각 스텝의 시간 간격 (초 단위)
        """
        self.dir_list = os.listdir(dataset_dir)
        self.dir_name = os.path.join("./"+dataset_dir)
        self.seq_length = seq_length
        self.future_step = future_step
        self.time_step = time_step
        
        # 예측 시간 계산 (초 단위)
        self.prediction_seconds = self.future_step * self.time_step
        
        # 스케일러 초기화
        self.temp_std_scaler = StandardScaler()
        self.co_std_scaler = StandardScaler()
        self.soot_std_scaler = StandardScaler()
        self.devc_std_scaler = StandardScaler()
        
        print(f"데이터셋 설정:")
        print(f"- 디렉토리 수: {len(self.dir_list)}")
        print(f"- 시퀀스 길이: {self.seq_length} 스텝")
        print(f"- 예측 스텝: {self.future_step} 스텝")
        print(f"- 시간 간격: {self.time_step} 초")
        print(f"- 예측 시간: {self.prediction_seconds} 초")
        
        # 스케일러 로드
        try:
            self.temp_std_scaler = joblib.load('./trained_scaler/temp_scaler.pkl')
            self.co_std_scaler = joblib.load('./trained_scaler/temp_scaler.pkl')
            self.soot_std_scaler = joblib.load('./trained_scaler/temp_scaler.pkl')
            self.devc_std_scaler = joblib.load('./trained_scaler/temp_scaler.pkl')
            print("스케일러 로드 완료")
        except:
            print("스케일러 파일을 찾을 수 없습니다. standard_scale() 메소드를 호출하여 스케일러를 학습시키세요.")
            # self.standard_scale()  # 스케일러 학습을 위해 주석 해제
        
        # 유효한 (폴더, 인덱스) 쌍 저장
        self.valid_samples = []
        
        for folder_idx, dir_name in enumerate(self.dir_list):
            folder_path = os.path.join(self.dir_name, dir_name)
            
            # 슬라이스 파일 목록 가져오기 및 인덱스 추출
            slice_files = glob.glob(os.path.join(folder_path, "slice_*.csv"))
            slice_indices = []
            
            for file_path in slice_files:
                try:
                    file_name = os.path.basename(file_path)
                    # slice_60_61.csv 형식의 파일명에서 인덱스 추출
                    parts = file_name.split('_')
                    if len(parts) >= 3 and parts[0] == "slice" and parts[1].isdigit():
                        slice_indices.append(int(parts[1]))
                except Exception as e:
                    print(f"슬라이스 파일명 {file_path} 파싱 오류: {e}")
            
            # 인덱스 정렬
            slice_indices = sorted(slice_indices)
            
            if not slice_indices:
                print(f"경고: {dir_name}에 유효한 슬라이스 파일이 없습니다.")
                continue
            
            # 센서 데이터 파일이 존재하는지 확인
            devc_file = os.path.join(folder_path, f"{dir_name}_devc.csv")
            if not os.path.exists(devc_file):
                print(f"경고: {dir_name}에 센서 데이터가 없습니다.")
                continue
                
            # 센서 데이터 파일의 행 수 확인
            try:
                devc_data = pd.read_csv(devc_file, header=1)
                if len(devc_data) < self.seq_length:
                    print(f"경고: {dir_name}의 센서 데이터가 시퀀스 길이({self.seq_length})보다 적습니다.")
                    continue
            except Exception as e:
                print(f"devc 파일 {devc_file} 읽기 오류: {e}")
                continue
            
            # 각 시작 인덱스에 대해 검증 (수정된 부분)
            # future_step과 seq_length를 고려하여 최대 가능한 인덱스 계산
            max_devc_idx = len(devc_data) - self.seq_length
            
            # 각 가능한 시작 인덱스에 대해 확인 (고정된 90이 아닌 실제 가능한 범위 사용)
            for idx in range(max_devc_idx + 1):
                label_idx = idx + self.future_step  # 예측 시점 인덱스
                
                # 해당 인덱스의 슬라이스 파일이 존재하는지 확인
                if label_idx in slice_indices:
                    self.valid_samples.append((folder_idx, idx))
                    
        print(f"유효한 샘플 수: {len(self.valid_samples)}")
        
        # 파라미터별 유효 샘플 수 영향 분석
        print(f"seq_length={self.seq_length}, future_step={self.future_step}일 때 유효 샘플 수: {len(self.valid_samples)}")
    
    def __len__(self):
        return len(self.valid_samples)
        
    def __getitem__(self, idx):
        if idx >= len(self.valid_samples):
            raise IndexError(f"인덱스 {idx}가 데이터셋 범위({len(self.valid_samples)})를 벗어났습니다.")
            
        folder_idx, sample_idx = self.valid_samples[idx]
        dir_name = self.dir_list[folder_idx]
        
        # 센서 데이터 로드 및 전처리
        try:
            data = pd.read_csv(os.path.join(f"{self.dir_name}/{dir_name}/", 
                                          f"{dir_name}_devc.csv"), header=1)
            data = data.values.astype(np.float32)
            data = data[:, 1:71]  # 첫 번째 열(시간) 제외하고 70개 센서 데이터 사용
            
            # 데이터 재구성
            re_data = []
            for row in data:
                row_2d = row.reshape(7, 10)
                row_2d[1:] = np.fliplr(row_2d[1:])
                row_2d[1:, [-2, -1]] = row_2d[1:, [-1, -2]]
                row_2d[1:, [-2, -3]] = row_2d[1:, [-3, -2]]
                transformed_row = row_2d.reshape(-1)
                re_data.append(transformed_row)
            
            data = np.array(re_data) 
            data = data[sample_idx:sample_idx+self.seq_length]
            # 센서 데이터 스케일링
            data = self.devc_std_scaler.transform(data)
        except Exception as e:
            print(f"{dir_name}의 인덱스 {sample_idx}에서 센서 데이터 처리 오류: {e}")
            # 재귀적으로 다른 샘플 시도 (깊이 제한을 위해 최대 5회)
            return self.__getitem__((idx + 1) % len(self))
        
        # 슬라이스 데이터 로드
        label_idx = sample_idx + self.future_step
        try:
            label_data = pd.read_csv(os.path.join(f"{self.dir_name}/{dir_name}/", 
                                               f"slice_{label_idx}_{label_idx + 1}.csv"), 
                                  sep=", ", 
                                  names=['X', 'Y', 'TEMPERATURE', 'SOOT_VISIBILITY', 'CO_FRACTION'],  
                                  engine="python",
                                  skiprows=2)
            
            # 각 변수별 개별 스케일링
            temp_values = label_data["TEMPERATURE"].values.astype(np.float32).reshape(-1, 1)
            co_values = label_data["CO_FRACTION"].values.astype(np.float32).reshape(-1, 1)
            soot_values = label_data["SOOT_VISIBILITY"].values.astype(np.float32).reshape(-1, 1)
            
            temp_scaled = self.temp_std_scaler.transform(temp_values).reshape(53, 66)
            co_scaled = self.co_std_scaler.transform(co_values).reshape(53, 66)
            soot_scaled = self.soot_std_scaler.transform(soot_values).reshape(53, 66)
            
            # 각 스케일링된 데이터 flip
            temp_scaled = np.flip(temp_scaled, axis=0)
            co_scaled = np.flip(co_scaled, axis=0)
            soot_scaled = np.flip(soot_scaled, axis=0)
            
            # 최종 데이터 형태로 재구성
            total_data = np.stack([temp_scaled, co_scaled, soot_scaled])
        except Exception as e:
            print(f"{dir_name}의 라벨 인덱스 {label_idx}에서 슬라이스 데이터 처리 오류: {e}")
            # 재귀적으로 다른 샘플 시도
            return self.__getitem__((idx + 1) % len(self))
        
        # 텐서 변환
        data = torch.tensor(data, dtype=torch.float32)
        total_data = torch.tensor(total_data, dtype=torch.float32)
        
        return data, total_data

    def standard_scale(self):
        """
        데이터셋의 모든 파일에서 데이터를 수집하고 스케일러를 학습시킨 후 저장
        """
        # 각 변수별 데이터 수집
        all_temp_data = []
        all_co_data = []
        all_soot_data = []
        all_devc_data = []
        
        # 각 폴더에 대해 반복
        for dir_name in self.dir_list:
            folder_path = os.path.join(self.dir_name, dir_name)
            
            # 센서 데이터 수집
            try:
                devc_file = os.path.join(folder_path, f"{dir_name}_devc.csv")
                devc_data = pd.read_csv(devc_file, header=1)
                devc_values = devc_data.iloc[:, 1:71].values.astype(np.float32)
                
                '''
                데이터 재구성 단계계
                이 부분은 pyrosim 온도계 설정 과정에서 순서를 제대로하지 않아 발생한
                실수를 다시 정상적인 데이터 순서로 만들기 위해 한 작업입니다.
                '''
                re_devc_data = []
                for row in devc_values:
                    row_2d = row.reshape(7, 10)
                    row_2d[1:] = np.fliplr(row_2d[1:])
                    row_2d[1:, [-2, -1]] = row_2d[1:, [-1, -2]]
                    row_2d[1:, [-2, -3]] = row_2d[1:, [-3, -2]]
                    transformed_row = row_2d.reshape(-1)
                    re_devc_data.append(transformed_row)
                
                all_devc_data.extend(re_devc_data)
            except Exception as e:
                print(f"devc 파일 {devc_file} 읽기 오류: {e}")
            
            # 슬라이스 파일 수집
            slice_files = glob.glob(os.path.join(folder_path, "slice_*.csv"))
            for file_path in slice_files:
                try:
                    slice_data = pd.read_csv(file_path, sep=", ",
                                        names=['X', 'Y', 'TEMPERATURE', 'SOOT_VISIBILITY', 'CO_FRACTION'],
                                        engine="python", skiprows=2)
                    
                    # 각 변수별 데이터 추출 및 수집
                    temp_values = slice_data['TEMPERATURE'].values.astype(np.float32).reshape(-1, 1)
                    co_values = slice_data['CO_FRACTION'].values.astype(np.float32).reshape(-1, 1)
                    soot_values = slice_data['SOOT_VISIBILITY'].values.astype(np.float32).reshape(-1, 1)
                    
                    all_temp_data.append(temp_values)
                    all_co_data.append(co_values)
                    all_soot_data.append(soot_values)
                    
                except Exception as e:
                    print(f"슬라이스 파일 {file_path} 읽기 오류: {e}")
        
        # 각 데이터 병합
        combined_temp_data = np.vstack(all_temp_data) if all_temp_data else np.array([]).reshape(0, 1)
        combined_co_data = np.vstack(all_co_data) if all_co_data else np.array([]).reshape(0, 1)
        combined_soot_data = np.vstack(all_soot_data) if all_soot_data else np.array([]).reshape(0, 1)
        combined_devc_data = np.vstack(all_devc_data) if all_devc_data else np.array([]).reshape(0, 70)
        
        # 각 스케일러 학습
        print(f"온도 스케일러 학습 ({len(combined_temp_data)}개 샘플)")
        self.temp_std_scaler.fit(combined_temp_data)
        
        print(f"CO 스케일러 학습 ({len(combined_co_data)}개 샘플)")
        self.co_std_scaler.fit(combined_co_data)
        
        print(f"그을음 스케일러 학습 ({len(combined_soot_data)}개 샘플)")
        self.soot_std_scaler.fit(combined_soot_data)
        
        print(f"센서 스케일러 학습 ({len(combined_devc_data)}개 샘플)")
        self.devc_std_scaler.fit(combined_devc_data)
        
        print("모든 스케일러 학습 완료!")

        # 스케일러 저장
        joblib.dump(self.temp_std_scaler, './trained_scaler/temp_scaler.pkl')
        joblib.dump(self.co_std_scaler, './trained_scaler/temp_scaler.pkl')
        joblib.dump(self.soot_std_scaler, './trained_scaler/temp_scaler.pkl')
        joblib.dump(self.devc_std_scaler, "./trained_scaler/temp_scaler.pkl")
        print("모든 스케일러 저장 완료!")



def visualize_images(true_image):
    # PyTorch 텐서를 NumPy 배열로 변환하여 시각화
    true_image_np = true_image.permute(2,0,1).cpu().numpy()  # (C, H, W) → (H, W, C)
    
    plt.imshow(true_image_np)  # 이미지를 출력
    plt.title("True Image")    # 그래프 제목 설정
    plt.axis("on")            # 축 숨김
    plt.show() 

if __name__ == "__main__":
    A=myDataset("./dataset")
    input_data,total_data = A[300]
    
    # heatmap(total_data,"kimmin")
    
        
