"""
공통 유틸리티 함수들
모든 MNIST 분류 모델에서 공통으로 사용
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import psutil
import json
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 전역 설정
RANDOM_STATE = 42
DATA_SIZE = 10000  # 모든 모델에서 동일하게 사용할 데이터 크기
TEST_SIZE = 0.2
RESULTS_DIR = "results"
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")

def ensure_directories():
    """결과 저장을 위한 디렉토리 생성"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

def load_common_mnist_data():
    """
    모든 모델에서 공통으로 사용할 MNIST 데이터 로드
    동일한 random_state와 데이터 크기 사용
    """
    print("공통 MNIST 데이터를 로딩중...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # 동일한 데이터 크기로 제한
    np.random.seed(RANDOM_STATE)
    indices = np.random.choice(len(X), size=DATA_SIZE, replace=False)
    X = X[indices]
    y = y[indices]
    
    print(f"데이터 크기: {X.shape}")
    print(f"클래스 분포: {np.bincount(y)}")
    
    return X, y

def preprocess_for_traditional_ml(X, y):
    """전통적인 머신러닝 모델을 위한 전처리"""
    # 픽셀 값을 0-1 범위로 정규화
    X = X / 255.0
    
    # 훈련/테스트 데이터 분할 (동일한 random_state 사용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def preprocess_for_cnn(X, y):
    """CNN을 위한 전처리"""
    # 이미지 형태로 변환
    X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """모니터링 시작"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print(f"{self.model_name} 성능 모니터링 시작...")
        
    def end_monitoring(self):
        """모니터링 종료"""
        self.end_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = current_memory
        
        training_time = self.end_time - self.start_time
        memory_used = self.peak_memory - self.start_memory
        
        print(f"{self.model_name} 성능 모니터링 완료!")
        print(f"훈련 시간: {training_time:.2f}초")
        print(f"메모리 사용량: {memory_used:.2f}MB")
        
        return {
            'model_name': self.model_name,
            'training_time': training_time,
            'memory_used': memory_used,
            'start_memory': self.start_memory,
            'peak_memory': self.peak_memory
        }

def save_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """혼동 행렬 시각화 및 저장"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, f'{model_name}_confusion_matrix.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 메모리 절약을 위해 창 닫기
    print(f"혼동 행렬 저장: {save_path}")

def save_classification_report(y_true, y_pred, model_name, save_path=None):
    """분류 보고서 저장"""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f'{model_name}_classification_report.json')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"분류 보고서 저장: {save_path}")
    return report

def save_performance_metrics(performance_data, save_path=None):
    """성능 메트릭 저장"""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"performance_metrics.json")
    
    # 기존 데이터 로드 (있다면)
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    
    # NumPy 타입을 Python 타입으로 변환
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # 새 데이터 추가 (NumPy 타입 변환)
    converted_data = convert_numpy_types(performance_data)
    existing_data[converted_data['model_name']] = {
        **converted_data,
        'timestamp': datetime.now().isoformat()
    }
    
    # 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"성능 메트릭 저장: {save_path}")

def evaluate_model_common(model, X_train, X_test, y_train, y_test, model_name):
    """
    공통 모델 평가 함수
    모든 모델에서 동일한 방식으로 평가
    """
    print(f"{model_name} 모델 평가중...")
    
    # 예측 시간 측정
    start_time = time.time()
    
    if hasattr(model, 'predict_proba'):
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)
    else:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_proba = None
    
    prediction_time = time.time() - start_time
    
    # 정확도 계산
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"훈련 정확도: {train_accuracy:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"예측 시간: {prediction_time:.4f}초")
    
    # 결과 저장
    save_confusion_matrix(y_test, test_pred, model_name)
    classification_report_data = save_classification_report(y_test, test_pred, model_name)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'prediction_time': prediction_time,
        'test_pred': test_pred,
        'test_proba': test_proba,
        'classification_report': classification_report_data
    }

def save_feature_importance(importance, model_name, feature_names=None, top_n=20):
    """특성 중요도 시각화 및 저장"""
    if feature_names is None:
        feature_names = [f'Pixel_{i}' for i in range(len(importance))]
    
    # 상위 N개 특성 선택
    top_indices = np.argsort(importance)[-top_n:]
    top_importance = importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_importance)
    plt.yticks(range(top_n), top_names)
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top {top_n} Feature Importance')
    
    save_path = os.path.join(IMAGES_DIR, f'{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"특성 중요도 저장: {save_path}")
    
    # 28x28 히트맵으로도 저장 (픽셀 데이터인 경우)
    if len(importance) == 784:  # MNIST 픽셀 수
        importance_image = importance.reshape(28, 28)
        plt.figure(figsize=(8, 6))
        plt.imshow(importance_image, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'{model_name} - Feature Importance Heatmap')
        
        heatmap_path = os.path.join(IMAGES_DIR, f'{model_name}_importance_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"중요도 히트맵 저장: {heatmap_path}")

def save_learning_curve(train_scores, val_scores, param_range, param_name, model_name):
    """학습 곡선 저장"""
    plt.figure(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    if param_name in ['C', 'gamma']:
        plt.xscale('log')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'{model_name} - Learning Curve ({param_name})')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(IMAGES_DIR, f'{model_name}_learning_curve_{param_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"학습 곡선 저장: {save_path}")

def create_summary_table(results_file=None):
    """모든 모델의 결과를 요약한 테이블 생성"""
    if results_file is None:
        results_file = os.path.join(RESULTS_DIR, "performance_metrics.json")
    
    if not os.path.exists(results_file):
        print("성능 메트릭 파일이 존재하지 않습니다.")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 테이블 데이터 생성
    summary_data = []
    for model_name, metrics in data.items():
        summary_data.append({
            'Model': model_name,
            'Test Accuracy': f"{metrics.get('test_accuracy', 0):.4f}",
            'Training Time (s)': f"{metrics.get('training_time', 0):.2f}",
            'Memory Used (MB)': f"{metrics.get('memory_used', 0):.2f}",
            'Prediction Time (s)': f"{metrics.get('prediction_time', 0):.4f}"
        })
    
    # 정확도 순으로 정렬
    summary_data.sort(key=lambda x: float(x['Test Accuracy']), reverse=True)
    
    return summary_data

def print_summary_table():
    """요약 테이블 출력"""
    summary_data = create_summary_table()
    if not summary_data:
        return
    
    print("\n" + "="*80)
    print("모든 모델 성능 요약")
    print("="*80)
    
    # 헤더 출력
    headers = list(summary_data[0].keys())
    print(f"{'Model':<20} {'Accuracy':<10} {'Train Time':<12} {'Memory':<12} {'Pred Time':<12}")
    print("-" * 80)
    
    # 데이터 출력
    for row in summary_data:
        print(f"{row['Model']:<20} {row['Test Accuracy']:<10} {row['Training Time (s)']:<12} "
              f"{row['Memory Used (MB)']:<12} {row['Prediction Time (s)']:<12}")
    
    print("="*80)

# 초기화
ensure_directories() 