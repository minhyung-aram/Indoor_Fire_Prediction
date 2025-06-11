import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ConvLSTM import *  # LSTM 모델 임포트
from processing_data2nd_test import *  # 데이터 처리 모듈 임포트
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import time
import math
from torchinfo import summary

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MAPE 계산 함수 (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE 계산 함수 (0으로 나누기 오류 방지 로직 포함)
    """
    # 0으로 나누기 방지를 위해 epsilon 값 설정
    epsilon = 1e-10
    # 절대값이 epsilon보다 작은 값들은 계산에서 제외
    mask = np.abs(y_true) > epsilon
    
    if not np.any(mask):
        return 0.0  # 모든 값이 너무 작으면 MAPE를 0으로 반환
    
    # 마스크를 적용하여 안전하게 계산
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100

# 예측 및 평가 함수
def predict_and_evaluate(model, dataloader):
    model.eval()
    
    # 스케일러 로드
    try:
        temp_scaler = joblib.load('temp_scaler.pkl')
        co_scaler = joblib.load('co_scaler.pkl')
        soot_scaler = joblib.load('soot_scaler.pkl')
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return
    
    # 예측값과 실제값 수집을 위한 리스트
    all_temp_preds = []
    all_co_preds = []
    all_soot_preds = []
    all_temp_targets = []
    all_co_targets = []
    all_soot_targets = []
    
    # 추론 시간 측정
    total_inference_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for idx, (data, label) in enumerate(tqdm(dataloader, desc="Predicting")):
            # 데이터 로드
            data, label = data.to(device), label.to(device).float()
            
            # 모델 예측 및 시간 측정
            start_time = time.time()
            temp, co, soot = model(data)
            end_time = time.time()
            
            # 추론 시간 기록
            inference_time = end_time - start_time
            total_inference_time += inference_time
            total_samples += data.size(0)
            
            # 예측값과 실제값 수집 (CPU로 이동)
            all_temp_preds.append(temp.cpu())
            all_co_preds.append(co.cpu())
            all_soot_preds.append(soot.cpu())
            all_temp_targets.append(label[:, 0, :, :].cpu())
            all_co_targets.append(label[:, 1, :, :].cpu())
            all_soot_targets.append(label[:, 2, :, :].cpu())
    
    # 샘플이 없을 경우 처리
    if total_samples == 0:
        print("Error: No samples processed!")
        return
    
    # 평균 추론 시간 계산
    avg_inference_time = total_inference_time / total_samples
    print(f"\n평균 추론 시간 (샘플당): {avg_inference_time:.4f} 초")
    print(f"총 처리 샘플 수: {total_samples}")
    
    # 전체 데이터에 대한 텐서 연결
    try:
        temp_preds = torch.cat(all_temp_preds, dim=0)
        co_preds = torch.cat(all_co_preds, dim=0)
        soot_preds = torch.cat(all_soot_preds, dim=0)
        
        temp_targets = torch.cat(all_temp_targets, dim=0)
        co_targets = torch.cat(all_co_targets, dim=0)
        soot_targets = torch.cat(all_soot_targets, dim=0)
    except RuntimeError as e:
        print(f"텐서 병합 오류: {e}")
        return
    
    # 예측 및 타깃을 NumPy 배열로 변환
    temp_preds_np = temp_preds.flatten().numpy()
    co_preds_np = co_preds.flatten().numpy()
    soot_preds_np = soot_preds.flatten().numpy()
    
    temp_targets_np = temp_targets.flatten().numpy()
    co_targets_np = co_targets.flatten().numpy()
    soot_targets_np = soot_targets.flatten().numpy()
    
    # NaN 및 무한값 확인 및 제거
    temp_mask = ~(np.isnan(temp_preds_np) | np.isinf(temp_preds_np) | np.isnan(temp_targets_np) | np.isinf(temp_targets_np))
    co_mask = ~(np.isnan(co_preds_np) | np.isinf(co_preds_np) | np.isnan(co_targets_np) | np.isinf(co_targets_np))
    soot_mask = ~(np.isnan(soot_preds_np) | np.isinf(soot_preds_np) | np.isnan(soot_targets_np) | np.isinf(soot_targets_np))
    
    temp_preds_np = temp_preds_np[temp_mask]
    temp_targets_np = temp_targets_np[temp_mask]
    co_preds_np = co_preds_np[co_mask]
    co_targets_np = co_targets_np[co_mask]
    soot_preds_np = soot_preds_np[soot_mask]
    soot_targets_np = soot_targets_np[soot_mask]
    
    # 원본 스케일로 변환 (역변환)
    try:
        temp_preds_inverted = temp_scaler.inverse_transform(temp_preds_np.reshape(-1, 1)).flatten()
        co_preds_inverted = co_scaler.inverse_transform(co_preds_np.reshape(-1, 1)).flatten()
        soot_preds_inverted = soot_scaler.inverse_transform(soot_preds_np.reshape(-1, 1)).flatten()
        
        temp_targets_inverted = temp_scaler.inverse_transform(temp_targets_np.reshape(-1, 1)).flatten()
        co_targets_inverted = co_scaler.inverse_transform(co_targets_np.reshape(-1, 1)).flatten()
        soot_targets_inverted = soot_scaler.inverse_transform(soot_targets_np.reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"역변환 오류: {e}")
        temp_preds_inverted = temp_preds_np
        co_preds_inverted = co_preds_np
        soot_preds_inverted = soot_preds_np
        temp_targets_inverted = temp_targets_np
        co_targets_inverted = co_targets_np
        soot_targets_inverted = soot_targets_np
    
    # 세 가지 주요 지표 계산
    # 1. R²
    temp_r2 = r2_score(temp_targets_inverted, temp_preds_inverted)
    co_r2 = r2_score(co_targets_inverted, co_preds_inverted)
    soot_r2 = r2_score(soot_targets_inverted, soot_preds_inverted)
    
    # 2. RMSE (Root Mean Square Error)
    temp_rmse = math.sqrt(mean_squared_error(temp_targets_np, temp_preds_np))
    co_rmse = math.sqrt(mean_squared_error(co_targets_np, co_preds_np))
    soot_rmse = math.sqrt(mean_squared_error(soot_targets_np, soot_preds_np))
    
    # 3. MAPE (Mean Absolute Percentage Error)
    # temp_mape = mean_absolute_percentage_error(temp_targets_np, temp_preds_np)
    # co_mape = mean_absolute_percentage_error(co_targets_np, co_preds_np)
    # soot_mape = mean_absolute_percentage_error(soot_targets_np, soot_preds_np) 
    
    # 결과 출력
    print("\n" + "="*40)
    print("평가 지표 결과")
    print("="*40)
    
    headers = ["변수", "R²", "RMSE"]
    print(f"{headers[0]:<12} {headers[1]:<10} {headers[2]:<10}")
    print("-"*35)
    print(f"Temperature  {temp_r2:<10.4f} {temp_rmse:<10.4f} ")
    print(f"CO           {co_r2:<10.4f} {co_rmse:<10.4f} ")
    print(f"Soot         {soot_r2:<10.4f} {soot_rmse:<10.4f}")
    print("-"*35)
    
    # 평균값 계산
    avg_r2 = np.mean([temp_r2, co_r2, soot_r2])
    avg_rmse = np.mean([temp_rmse, co_rmse, soot_rmse])
    # avg_mape = np.mean([temp_mape, co_mape, soot_mape])
    print(f"평균         {avg_r2:<10.4f} {avg_rmse:<10.4f}")
    print("="*40)


# 테스트 실행
if __name__ == "__main__":
    print(f"사용 장치: {device}")
    
    # 배치 크기 설정
    batch_size = 64
    
    # 테스트 데이터셋 로드
    try:
        dataset = myDataset("test_dataset")
        print(f"데이터셋 크기: {len(dataset)}")
        
        # worker 수 조정
        num_workers = min(4, os.cpu_count() or 1)
        test_dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,  # 재현성을 위해 shuffle=False 설정
            num_workers=num_workers, 
            pin_memory=True
        )
    except Exception as e:
        print(f"데이터셋 로드 오류: {e}")
        exit(1)
    
    # 모델 로드
    try:
        model = TemperatureConvLSTM().to(device)
        checkpoint_path = "./ConvLSTM_checkpoints/conv_lstm.pth"
    
        if os.path.exists(checkpoint_path):
            print(f"모델 로드 중: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            
            # checkpoint 구조 확인
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            print(summary(model))
        else:
            print("체크포인트를 찾을 수 없습니다. 종료합니다.")
            exit(1)
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        exit(1)
    
    # 예측 및 평가 실행
    predict_and_evaluate(model, test_dataloader)