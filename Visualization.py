import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from lstm_model import TemperatureLSTM  # 모델 임포트
from processing_data2nd_test import myDataset  # 데이터셋 임포트
import joblib
import time
import os
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

'''
모델이 예측한 분포를 시각화하는 코드입니다.
모든 테스트 데이터 셋에대한 예측을 쭉 보거나
특정 시나리오를 인덱스로 쭉 직접 보거나
아무 인덱스를 입력하여 랜덤으로 볼수도 있습니다.
'''
# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_scenario_range(model, dataset_path, scenario_name, start_idx=None, end_idx=None, use_inverse_transform=True):
    """
    특정 시나리오(폴더)의 특정 인덱스 범위의 결과를 시각화합니다.
    
    Args:
        model: 학습된 모델
        dataset_path: 데이터셋 경로
        scenario_name: 시각화할 시나리오(폴더) 이름
        start_idx: 시각화할 시작 인덱스 (None이면 처음부터)
        end_idx: 시각화할 끝 인덱스 (None이면 마지막까지)
        use_inverse_transform: 스케일러 역변환 사용 여부
    """
    print(f"시나리오 '{scenario_name}' 시각화 중...")
    
    # 특정 시나리오의 유효한 샘플 정보 찾기
    scenario_samples = []
    
    # 원래 데이터셋 로드
    full_dataset = myDataset(dataset_path)
    
    # 전체 유효 샘플 중에서 특정 시나리오에 해당하는 정보 찾기
    for idx in range(len(full_dataset)):
        folder_idx, sample_idx = full_dataset.valid_samples[idx]
        folder_name = full_dataset.dir_list[folder_idx]
        
        if folder_name == scenario_name:
            scenario_samples.append((idx, sample_idx))  # (데이터셋 인덱스, 원본 인덱스) 저장
    
    if not scenario_samples:
        print(f"시나리오 '{scenario_name}'에 해당하는 샘플이 없습니다.")
        return
    
    # 시나리오에 있는 원본 인덱스 범위 출력
    orig_indices = [item[1] for item in scenario_samples]
    min_idx = min(orig_indices)
    max_idx = max(orig_indices)
    print(f"시나리오 '{scenario_name}'에서 {len(scenario_samples)}개의 샘플을 찾았습니다.")
    print(f"원본 인덱스 범위: {min_idx} ~ {max_idx}")
    
    # 시작 인덱스와 끝 인덱스 설정
    if start_idx is None:
        start_idx = min_idx
    
    if end_idx is None:
        end_idx = max_idx
    
    # 인덱스 범위 필터링
    filtered_samples = []
    for dataset_idx, orig_idx in scenario_samples:
        if start_idx <= orig_idx <= end_idx:
            filtered_samples.append(dataset_idx)
    
    if not filtered_samples:
        print(f"지정한 인덱스 범위 ({start_idx}~{end_idx})에 해당하는 샘플이 없습니다.")
        return
    
    print(f"인덱스 범위 {start_idx}~{end_idx}에서 {len(filtered_samples)}개의 샘플을 시각화합니다.")
    
    # 모델 평가 모드 설정
    model.eval()
    
    # 각 샘플에 대해 시각화
    with torch.no_grad():
        for dataset_idx in filtered_samples:
            # 데이터 로드
            input_data, target_data = full_dataset[dataset_idx]
            folder_idx, orig_idx = full_dataset.valid_samples[dataset_idx]
            
            # 배치 차원 추가
            input_batch = input_data.unsqueeze(0).to(device)
            
            # 모델 예측
            start_time = time.time()
            temp_pred, co_pred, soot_pred = model(input_batch)
            inference_time = time.time() - start_time
            
            # 예측 결과 합치기
            pred_combined = torch.stack([temp_pred[0], co_pred[0], soot_pred[0]], dim=0)
            
            # 시각화 제목
            title = f"시나리오: {scenario_name}, 인덱스: {orig_idx}, 시간: {inference_time:.4f}s"
            
            # 실제값과 예측값 비교 시각화
            compare_true_pred(
                target_data, 
                pred_combined, 
                title, 
                pause=True,
                use_inverse_transform=use_inverse_transform
            )

def inverse_transform_data(data, scaler_name):
    """
    데이터를 스케일러로 역변환합니다.
    
    Args:
        data: 변환할 데이터 (2D 배열 또는 텐서)
        scaler_name: 스케일러 이름 ('temp', 'co', 'soot')
    
    Returns:
        역변환된 데이터
    """
    # PyTorch 텐서인 경우 NumPy로 변환
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # 원본 형태 저장
    original_shape = data.shape
    
    try:
        # 스케일러 로드
        if scaler_name == 'temp':
            scaler = joblib.load('temp_scaler.pkl')
        elif scaler_name == 'co':
            scaler = joblib.load('co_scaler.pkl')
        elif scaler_name == 'soot':
            scaler = joblib.load('soot_scaler.pkl')
        else:
            raise ValueError(f"유효하지 않은 스케일러 이름: {scaler_name}")
        
        # 2D 데이터를 1D로 변환하여 역변환
        flattened_data = data.reshape(-1, 1)
        inverted_data = scaler.inverse_transform(flattened_data)
        
        # 원본 형태로 복원
        return inverted_data.reshape(original_shape)
    
    except Exception as e:
        print(f"역변환 오류 ({scaler_name}): {e}")
        print("원본 데이터를 그대로 반환합니다.")
        return data

def visualize_heatmaps(data_matrix, title="Data Visualization", use_inverse_transform=True):
    """
    온도, CO, Soot 데이터를 한 창에 3개의 서브플롯으로 시각화합니다.
    
    Args:
        data_matrix: [3, H, W] 형태의 데이터 (PyTorch 텐서 또는 NumPy 배열)
        title: 그래프 전체 제목
        use_inverse_transform: 스케일러 역변환 사용 여부
    """
    # PyTorch 텐서인 경우 NumPy로 변환
    if isinstance(data_matrix, torch.Tensor):
        data_matrix = data_matrix.cpu().numpy()
    
    # 3채널(온도, CO, Soot)이 맞는지 확인
    if data_matrix.shape[0] != 3:
        raise ValueError(f"데이터는 [3, H, W] 형태여야 합니다. 현재 형태: {data_matrix.shape}")
    
    # 서브플롯 생성
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 각 변수별 이름, 컬러맵, 스케일러 설정
    variables = [
        ("Temperature", "hot", "temp"),
        ("CO Fraction", "plasma", "co"),
        ("Soot Visibility", "inferno", "soot")
    ]
    
    # 각 변수별 시각화
    for i, (var_name, cmap, scaler_name) in enumerate(variables):
        # 스케일러 역변환 (필요 시)
        if use_inverse_transform:
            try:
                display_data = inverse_transform_data(data_matrix[i], scaler_name)
                var_title = f"{var_name} (Original Scale)"
            except Exception as e:
                print(f"역변환 실패: {e}")
                display_data = data_matrix[i]
                var_title = f"{var_name} (Normalized)"
        else:
            display_data = data_matrix[i]
            var_title = f"{var_name} (Normalized)"
        
        im = axes[i].imshow(display_data, cmap=cmap)
        axes[i].set_title(var_title)
        plt.colorbar(im, ax=axes[i])
        axes[i].axis('on')
    
    # 전체 제목 설정 및 레이아웃 조정
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 제목을 위한 공간 확보
    
    return fig  # Figure 객체 반환

def compare_true_pred(true_data, pred_data, title="True vs Predicted", pause=True, use_inverse_transform=True):
    """
    실제값과 예측값을 비교하여 시각화합니다.
    간격이 늘어난 히트맵 레이아웃으로 표시합니다.
    
    Args:
        true_data: [3, H, W] 형태의 실제 데이터
        pred_data: [3, H, W] 형태의 예측 데이터
        title: 그래프 전체 제목
        pause: 사용자 입력 대기 여부
        use_inverse_transform: 스케일러 역변환 사용 여부
    """
    # PyTorch 텐서인 경우 NumPy로 변환
    if isinstance(true_data, torch.Tensor):
        true_data = true_data.cpu().numpy()
    if isinstance(pred_data, torch.Tensor):
        pred_data = pred_data.cpu().numpy()
    
    # 각 데이터가 3채널(온도, CO, Soot)인지 확인
    if true_data.shape[0] != 3 or pred_data.shape[0] != 3:
        raise ValueError("데이터는 [3, H, W] 형태여야 합니다.")
    
    # 서브플롯 생성 (3행 2열) - 더 큰 여백 설정
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    
    # 서브플롯 간 간격 조정 - 행 간격 확대
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 각 변수별 이름, 컬러맵, 스케일러 설정
    variables = [
        ("Temperature", "hot", "temp"),
        ("CO Fraction", "plasma", "co"),
        ("Soot Visibility", "inferno", "soot")
    ]
    
    # 각 변수별 실제값과 예측값 시각화
    for i, (var_name, cmap, scaler_name) in enumerate(variables):
        # 스케일러 역변환 (필요 시)
        if use_inverse_transform:
            try:
                true_display = inverse_transform_data(true_data[i], scaler_name)
                pred_display = inverse_transform_data(pred_data[i], scaler_name)
                unit_suffix = get_unit_for_variable(var_name)
                true_title = f"{var_name} (True) {unit_suffix}"
                pred_title = f"{var_name} (Predicted) {unit_suffix}"
            except Exception as e:
                print(f"역변환 실패: {e}")
                true_display = true_data[i]
                pred_display = pred_data[i]
                true_title = f"{var_name} (True, Norm.)"
                pred_title = f"{var_name} (Predicted, Norm.)"
        else:
            true_display = true_data[i]
            pred_display = pred_data[i]
            true_title = f"{var_name} (True, Norm.)"
            pred_title = f"{var_name} (Predicted, Norm.)"
        
        # 실제값과 예측값의 값 범위를 동일하게 설정
        vmin = min(true_display.min(), pred_display.min())
        vmax = max(true_display.max(), pred_display.max())
        
        # 실제값
        im1 = axes[i, 0].imshow(true_display, cmap=cmap, vmin=vmin, vmax=vmax)
        #axes[i, 0].set_title(true_title, fontsize=12, pad=10)  # 제목과 그래프 사이 간격 늘림
        cbar1 = plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)  # 컬러바 크기 및 위치 조정
        
        # 예측값
        im2 = axes[i, 1].imshow(pred_display, cmap=cmap, vmin=vmin, vmax=vmax)
       # axes[i, 1].set_title(pred_title, fontsize=12, pad=10)  # 제목과 그래프 사이 간격 늘림
        cbar2 = plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)  # 컬러바 크기 및 위치 조정
    
    # 전체 제목 설정 및 레이아웃 조정
    scale_text = "Original Scale" if use_inverse_transform else "Normalized Scale"
    plt.suptitle(f"{title} ({scale_text})", fontsize=16, y=0.98)
    
    # 기존 tight_layout 대신 수동으로 여백 조정
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9)
    
    # 그래프 표시
    plt.show(block=False)
    
    # 사용자 입력 대기 (필요 시)
    if pause:
        input("아무 키나 눌러서 다음 이미지로 진행하세요...")
    
    return fig  # Figure 객체 반환

def visualize_heatmaps(data_matrix, title="Data Visualization", use_inverse_transform=True):
    """
    온도, CO, Soot 데이터를 한 창에 3개의 서브플롯으로 시각화합니다.
    간격이 늘어난 히트맵 레이아웃으로 표시합니다.
    
    Args:
        data_matrix: [3, H, W] 형태의 데이터 (PyTorch 텐서 또는 NumPy 배열)
        title: 그래프 전체 제목
        use_inverse_transform: 스케일러 역변환 사용 여부
    """
    # PyTorch 텐서인 경우 NumPy로 변환
    if isinstance(data_matrix, torch.Tensor):
        data_matrix = data_matrix.cpu().numpy()
    
    # 3채널(온도, CO, Soot)이 맞는지 확인
    if data_matrix.shape[0] != 3:
        raise ValueError(f"데이터는 [3, H, W] 형태여야 합니다. 현재 형태: {data_matrix.shape}")
    
    # 서브플롯 생성 - 더 큰 여백 설정
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 서브플롯 간 간격 확대
    plt.subplots_adjust(wspace=0.4)  # 가로 간격 확대
    
    # 각 변수별 이름, 컬러맵, 스케일러 설정
    variables = [
        ("Temperature", "hot", "temp"),
        ("CO Fraction", "plasma", "co"),
        ("Soot Visibility", "inferno", "soot")
    ]
    
    # 각 변수별 시각화
    for i, (var_name, cmap, scaler_name) in enumerate(variables):
        # 스케일러 역변환 (필요 시)
        if use_inverse_transform:
            try:
                display_data = inverse_transform_data(data_matrix[i], scaler_name)
                unit_suffix = get_unit_for_variable(var_name)
                var_title = f"{var_name} {unit_suffix}"
            except Exception as e:
                print(f"역변환 실패: {e}")
                display_data = data_matrix[i]
                var_title = f"{var_name} (Normalized)"
        else:
            display_data = data_matrix[i]
            var_title = f"{var_name} (Normalized)"
        
        im = axes[i].imshow(display_data, cmap=cmap)
        axes[i].set_title(var_title, fontsize=12, pad=10)  # 제목과 그래프 사이 간격 늘림
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)  # 컬러바 크기 및 위치 조정
        axes[i].axis('on')
    
    # 전체 제목 설정
    plt.suptitle(title, fontsize=16, y=0.98)
    
    # 기존 tight_layout 대신 수동으로 여백 조정
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    
    return fig  # Figure 객체 반환

def get_unit_for_variable(var_name):
    """변수명에 따른 단위 반환"""
    if "Temperature" in var_name:
        return "(°C)"
    elif "CO" in var_name:
        return "(ppm)"
    elif "Soot" in var_name:
        return "(m)"
    else:
        return ""

def visualize_model_predictions(model, dataloader, num_samples=None, save_path=None, use_inverse_transform=True):
    """
    모델의 예측 결과를 시각화합니다.
    
    Args:
        model: 예측에 사용할 모델
        dataloader: 데이터 로더
        num_samples: 시각화할 샘플 수 (None이면 전체)
        save_path: 결과 저장 경로 (None이면 저장하지 않음)
        use_inverse_transform: 스케일러 역변환 사용 여부
    """
    model.eval()  # 평가 모드로 설정
    
    # 저장 경로 설정
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(tqdm(dataloader, desc="Visualizing")):
            # 데이터 디바이스 이동
            data, label = data.to(device), label.to(device)
            
            # 모델 예측
            start_time = time.time()
            temp_pred, co_pred, soot_pred = model(data)
            inference_time = time.time() - start_time
            
            # 예측 결과 합치기
            pred_combined = torch.stack([temp_pred, co_pred, soot_pred], dim=1)
            
            # 배치 내 각 샘플에 대해 시각화
            for i in range(data.size(0)):
                # 샘플 수 제한 확인
                if num_samples is not None and sample_count >= num_samples:
                    return
                
                # 현재 샘플 추출
                true_sample = label[i]
                pred_sample = pred_combined[i]
                
                # 시각화 제목
                title = f"Sample {sample_count} - Inference: {inference_time:.4f}s"
                
                # 실제값과 예측값 비교 시각화
                fig = compare_true_pred(
                    true_sample, 
                    pred_sample, 
                    title, 
                    pause=(save_path is None),
                    use_inverse_transform=use_inverse_transform
                )
                
                # 결과 저장 (필요 시)
                if save_path:
                    scale_text = "original" if use_inverse_transform else "normalized"
                    fig_path = os.path.join(save_path, f"sample_{sample_count:04d}_{scale_text}.png")
                    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                sample_count += 1

def visualize_specific_scenario(model, dataset_path, scenario_name, use_inverse_transform=True):
    """
    특정 시나리오(폴더)의 결과만 시각화합니다.
    
    Args:
        model: 학습된 모델
        dataset_path: 데이터셋 경로
        scenario_name: 시각화할 시나리오(폴더) 이름
        use_inverse_transform: 스케일러 역변환 사용 여부
    """
    print(f"시나리오 '{scenario_name}' 시각화 중...")
    
    # 특정 시나리오의 유효한 샘플 인덱스 찾기
    found_indices = []
    
    # 원래 데이터셋 로드
    full_dataset = myDataset(dataset_path)
    
    # 전체 유효 샘플 중에서 특정 시나리오에 해당하는 인덱스 찾기
    for idx in range(len(full_dataset)):
        folder_idx, sample_idx = full_dataset.valid_samples[idx]
        folder_name = full_dataset.dir_list[folder_idx]
        
        if folder_name == scenario_name:
            found_indices.append(idx)
    
    if not found_indices:
        print(f"시나리오 '{scenario_name}'에 해당하는 샘플이 없습니다.")
        return
    
    print(f"시나리오 '{scenario_name}'에서 {len(found_indices)}개의 샘플을 찾았습니다.")
    
    # 모델 평가 모드 설정
    model.eval()
    
    # 각 샘플에 대해 시각화
    with torch.no_grad():
        for sample_idx in found_indices:
            # 데이터 로드
            input_data, target_data = full_dataset[sample_idx]
            folder_idx, orig_idx = full_dataset.valid_samples[sample_idx]
            
            # 배치 차원 추가
            input_batch = input_data.unsqueeze(0).to(device)
            
            # 모델 예측
            start_time = time.time()
            temp_pred, co_pred, soot_pred = model(input_batch)
            inference_time = time.time() - start_time
            
            # 예측 결과 합치기
            pred_combined = torch.stack([temp_pred[0], co_pred[0], soot_pred[0]], dim=0)
            
            # 시각화 제목
            title = f"시나리오: {scenario_name}, 인덱스: {orig_idx}, 시간: {inference_time:.4f}s"
            
            # 실제값과 예측값 비교 시각화
            compare_true_pred(
                target_data, 
                pred_combined, 
                title, 
                pause=True,
                use_inverse_transform=use_inverse_transform
            )

# 테스트 실행
if __name__ == "__main__":
    print(f"사용 장치: {device}")
    
    # 스케일러 역변환 사용 여부
    use_inverse_transform = True  # True: 원본 스케일, False: 정규화된 스케일
    
    # 모델 로드
    try:
        model = TemperatureLSTM().to(device)
        checkpoint_path = "./lstm1_100.pth"
        
        if os.path.exists(checkpoint_path):
            print(f"모델 로드 중: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # checkpoint 구조 확인
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
            print("모델 로드 완료")
        else:
            print("체크포인트를 찾을 수 없습니다. 종료합니다.")
            exit(1)
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        exit(1)
    
    # 시각화 모드 선택
    print("\n시각화 모드를 선택하세요:")
    print("1. 전체 데이터셋에서 일부 샘플 시각화")
    print("2. 특정 시나리오(폴더) 전체 시각화")
    print("3. 특정 인덱스 샘플 시각화")
    print("4. 특정 시나리오의 인덱스 범위 시각화")
    
    choice = input("선택 (1-4): ")
    
    dataset_path = "test_dataset" if os.path.exists("test_dataset") else "dataset"
    
    if choice == '1':
        # 일반 데이터셋 로드
        try:
            dataset = myDataset(dataset_path)
            print(f"데이터셋 크기: {len(dataset)}")
            
            # DataLoader 설정
            test_dataloader = DataLoader(
                dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == 'cuda')
            )
            
            # 샘플 수 입력
            num_samples = int(input("시각화할 샘플 수: "))
            
            # 모델 예측 및 시각화
            print("\n모델 예측 결과 시각화 시작...")
            visualize_model_predictions(
                model=model,
                dataloader=test_dataloader,
                num_samples=num_samples,
                save_path=None,
                use_inverse_transform=use_inverse_transform
            )
        except Exception as e:
            print(f"오류 발생: {e}")
            
    elif choice == '2':
        # 특정 시나리오 전체 시각화
        try:
            # 사용 가능한 시나리오 목록 표시
            available_scenarios = sorted(os.listdir(dataset_path))
            
            print("\n사용 가능한 시나리오 목록:")
            for i, scenario in enumerate(available_scenarios):
                print(f"{i+1}. {scenario}")
            
            scenario_idx = int(input("\n시각화할 시나리오 번호: ")) - 1
            
            if 0 <= scenario_idx < len(available_scenarios):
                selected_scenario = available_scenarios[scenario_idx]
                print(f"선택된 시나리오: {selected_scenario}")
                
                visualize_specific_scenario(
                    model=model,
                    dataset_path=dataset_path,
                    scenario_name=selected_scenario,
                    use_inverse_transform=use_inverse_transform
                )
            else:
                print("유효하지 않은 시나리오 번호입니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
            
    elif choice == '3':
        # 특정 인덱스 샘플 시각화
        try:
            dataset = myDataset(dataset_path)
            print(f"데이터셋 크기: {len(dataset)}")
            
            index = int(input(f"시각화할 샘플 인덱스 (0-{len(dataset)-1}): "))
            
            if 0 <= index < len(dataset):
                # 데이터 로드
                input_data, target_data = dataset[index]
                
                # 폴더 정보 얻기
                folder_idx, sample_idx = dataset.valid_samples[index]
                folder_name = dataset.dir_list[folder_idx]
                
                # 배치 차원 추가
                input_batch = input_data.unsqueeze(0).to(device)
                
                # 모델 예측
                with torch.no_grad():
                    start_time = time.time()
                    temp_pred, co_pred, soot_pred = model(input_batch)
                    inference_time = time.time() - start_time
                
                # 예측 결과 합치기
                pred_combined = torch.stack([temp_pred[0], co_pred[0], soot_pred[0]], dim=0)
                
                # 시각화 제목
                title = f"Folder: {folder_name}, idx: {sample_idx}, time: {inference_time:.4f}s"
                
                # 실제값과 예측값 비교 시각화
                compare_true_pred(
                    target_data, 
                    pred_combined, 
                    title, 
                    pause=True,
                    use_inverse_transform=use_inverse_transform
                )
            else:
                print("유효하지 않은 인덱스입니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
    
    elif choice == '4':
        # 특정 시나리오의 인덱스 범위 시각화
        try:
            # 사용 가능한 시나리오 목록 표시
            available_scenarios = sorted(os.listdir(dataset_path))
            
            print("\n사용 가능한 시나리오 목록:")
            for i, scenario in enumerate(available_scenarios):
                print(f"{i+1}. {scenario}")
            
            scenario_idx = int(input("\n시각화할 시나리오 번호: ")) - 1
            
            if 0 <= scenario_idx < len(available_scenarios):
                selected_scenario = available_scenarios[scenario_idx]
                print(f"선택된 시나리오: {selected_scenario}")
                
                # 데이터셋 로드하여 해당 시나리오의 인덱스 범위 확인
                full_dataset = myDataset(dataset_path)
                orig_indices = []
                
                for idx in range(len(full_dataset)):
                    folder_idx, sample_idx = full_dataset.valid_samples[idx]
                    folder_name = full_dataset.dir_list[folder_idx]
                    
                    if folder_name == selected_scenario:
                        orig_indices.append(sample_idx)
                
                if not orig_indices:
                    print(f"시나리오 '{selected_scenario}'에 해당하는 샘플이 없습니다.")
                
                # 시나리오에 있는 원본 인덱스 범위 출력
                min_idx = min(orig_indices)
                max_idx = max(orig_indices)
                print(f"사용 가능한 인덱스 범위: {min_idx} ~ {max_idx}")
                
                # 인덱스 범위 입력 받기
                start_idx = input(f"시작 인덱스 ({min_idx}~{max_idx}, 기본값: {min_idx}): ")
                start_idx = int(start_idx) if start_idx.strip() else min_idx
                
                end_idx = input(f"끝 인덱스 ({start_idx}~{max_idx}, 기본값: {max_idx}): ")
                end_idx = int(end_idx) if end_idx.strip() else max_idx
                
                # 유효성 검사
                if start_idx < min_idx:
                    start_idx = min_idx
                    print(f"시작 인덱스가 범위보다 작아 {min_idx}로 조정됩니다.")
                
                if end_idx > max_idx:
                    end_idx = max_idx
                    print(f"끝 인덱스가 범위보다 커서 {max_idx}로 조정됩니다.")
                
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                    print("시작 인덱스가 끝 인덱스보다 커서 두 값을 교환합니다.")
                
                # 범위 시각화 실행
                visualize_scenario_range(
                    model=model,
                    dataset_path=dataset_path,
                    scenario_name=selected_scenario,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    use_inverse_transform=use_inverse_transform
                )
            else:
                print("유효하지 않은 시나리오 번호입니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
    
    else:
        print("유효하지 않은 선택입니다.")
