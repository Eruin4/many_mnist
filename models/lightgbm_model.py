"""
LightGBM을 이용한 MNIST 손글씨 분류 (간단 버전)
기본 모델만 실행하여 빠르게 결과를 얻습니다.
"""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_common_mnist_data, preprocess_for_traditional_ml, 
    PerformanceMonitor, evaluate_model_common, save_performance_metrics,
    IMAGES_DIR
)
import os

def lightgbm_simple_analysis():
    """LightGBM 간단 분석 실행"""
    print("=" * 60)
    print("LightGBM MNIST 분류 분석 (간단 버전)")
    print("=" * 60)
    
    # 1. 공통 데이터 로드
    X, y = load_common_mnist_data()
    X_train, X_test, y_train, y_test = preprocess_for_traditional_ml(X, y)
    
    # 2. 성능 모니터링 시작
    monitor = PerformanceMonitor("LightGBM")
    monitor.start_monitoring()
    
    # 3. 기본 LightGBM 모델
    print("\n1. 기본 LightGBM 모델 훈련")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # 기본 모델 평가
    results = evaluate_model_common(lgb_model, X_train, X_test, y_train, y_test, "LightGBM_Basic")
    
    # 4. 간단한 특성 중요도 분석
    print("\n2. 특성 중요도 분석")
    analyze_feature_importance_simple(lgb_model)
    
    # 5. 성능 모니터링 종료 및 저장
    performance_data = monitor.end_monitoring()
    
    # 결과 데이터에 추가 정보 포함
    performance_data.update({
        'test_accuracy': results['test_accuracy'],
        'train_accuracy': results['train_accuracy'], 
        'prediction_time': results['prediction_time'],
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_features': X_train.shape[1]
    })
    
    save_performance_metrics(performance_data)
    
    print("\n3. LightGBM 분석 완료!")
    print(f"기본 모델 정확도: {results['test_accuracy']:.4f}")
    print(f"모든 결과는 results/ 폴더에 저장되었습니다.")
    
    return lgb_model, performance_data

def analyze_feature_importance_simple(lgb_model):
    """간단한 특성 중요도 분석"""
    # 특성 중요도 가져오기
    importance_scores = lgb_model.feature_importances_
    
    # 특성 중요도를 28x28 이미지로 시각화
    importance_image = importance_scores.reshape(28, 28)
    plt.figure(figsize=(8, 6))
    plt.imshow(importance_image, cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.title('LightGBM - Feature Importance Heatmap (28x28)')
    
    save_path = os.path.join(IMAGES_DIR, 'LightGBM_importance_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"중요도 히트맵 저장: {save_path}")

if __name__ == "__main__":
    model, performance = lightgbm_simple_analysis() 