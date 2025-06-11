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
    def __init__(self, dataset_dir, seq_length=30):
        self.dir_list = os.listdir(dataset_dir)
        self.dir_name = os.path.join("./"+dataset_dir)
        self.temp_std_scaler = StandardScaler()
        self.co_std_scaler = StandardScaler()  # 오타 수정: co_std_sclaer -> co_std_scaler
        self.soot_std_scaler = StandardScaler()
        self.devc_std_scaler = StandardScaler()
        print(self.dir_list)
        # self.standard_scale()
        self.temp_std_scaler = joblib.load('temp_scaler.pkl')
        self.co_std_scaler = joblib.load('co_scaler.pkl')
        self.soot_std_scaler = joblib.load('soot_scaler.pkl')
        self.devc_std_scaler = joblib.load('devc_scaler.pkl')

    def __len__(self):
        return len(self.dir_list)*90
        
    def __getitem__(self, idx):      
        self.file_num = idx//90
        self.idx = idx%90
        
        # 센서 데이터 로드 및 전처리
        data = pd.read_csv(os.path.join(f"{self.dir_name}/{self.dir_list[self.file_num]}/", 
                                      self.dir_list[self.file_num] + "_devc.csv"), header=1)
        data = data.values.astype(np.float32)
        data = data[:,1:71]
        
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
        data = data[self.idx:self.idx+30]
        # 센서 데이터 스케일링
        data = self.devc_std_scaler.transform(data)
        
        # 슬라이스 데이터 로드
        label_idx = self.idx + 60
        label_data = pd.read_csv(os.path.join(f"{self.dir_name}/{self.dir_list[self.file_num]}/", 
                                            f"slice_{label_idx}_{label_idx + 1}.csv"), sep=", ", 
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
        
        # 텐서 변환
        data = torch.tensor(data, dtype=torch.float32)
        # 0부터 19 사이의 무작위 정수 생성


        
        total_data = torch.tensor(total_data, dtype=torch.float32)
        
        return data, total_data   

    def standard_scale(self):
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
                
                # 데이터 재구성
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
                print(f"Error reading devc file {devc_file}: {e}")
            
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
                    print(f"Error reading slice file {file_path}: {e}")
        
        # 각 데이터 병합
        combined_temp_data = np.vstack(all_temp_data) if all_temp_data else np.array([]).reshape(0, 1)
        combined_co_data = np.vstack(all_co_data) if all_co_data else np.array([]).reshape(0, 1)
        combined_soot_data = np.vstack(all_soot_data) if all_soot_data else np.array([]).reshape(0, 1)
        combined_devc_data = np.vstack(all_devc_data) if all_devc_data else np.array([]).reshape(0, 70)
        
        # 각 스케일러 학습
        print(f"Fitting temp scaler with {len(combined_temp_data)} samples")
        self.temp_std_scaler.fit(combined_temp_data)
        
        print(f"Fitting CO scaler with {len(combined_co_data)} samples")
        self.co_std_scaler.fit(combined_co_data)
        
        print(f"Fitting soot scaler with {len(combined_soot_data)} samples")
        self.soot_std_scaler.fit(combined_soot_data)
        
        print(f"Fitting devc scaler with {len(combined_devc_data)} samples")
        self.devc_std_scaler.fit(combined_devc_data)
        
        print("All scalers fitted successfully!")

        joblib.dump(self.temp_std_scaler, 'temp_scaler.pkl')
        joblib.dump(self.co_std_scaler, 'co_scaler.pkl')
        joblib.dump(self.soot_std_scaler, 'soot_scaler.pkl')
        joblib.dump(self.devc_std_scaler, "devc_scaler.pkl")



        
        

# def heatmap(data_matrix,title):
#     plt.figure(figsize=(12, 8))
#     heatmap = sns.heatmap(data_matrix[2],cmap="jet")
#     plt.title(title)
#     # plt.gca().invert_yaxis()
#     plt.savefig('fig1.png')



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
    print(input_data)
    
    # heatmap(total_data,"kimmin")
    
        
