"""
XGBoost를 이용한 MNIST 손글씨 분류
통일된 데이터셋과 성능 모니터링 적용
"""

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost를 설치해주세요: pip install xgboost")
    import sys
    sys.exit(1)
    
from sklearn.model_selection import validation_curve, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_common_mnist_data, preprocess_for_traditional_ml, 
    PerformanceMonitor, evaluate_model_common, save_performance_metrics,
    save_learning_curve, IMAGES_DIR
)
import os

def xgboost_analysis():
    """XGBoost 분석 실행"""
    print("=" * 60)
    print("XGBoost MNIST 분류 분석 (통일된 데이터셋)")
    print("=" * 60)
    
    # 1. 공통 데이터 로드
    X, y = load_common_mnist_data()
    X_train, X_test, y_train, y_test = preprocess_for_traditional_ml(X, y)
    
    # 2. 성능 모니터링 시작
    monitor = PerformanceMonitor("XGBoost")
    monitor.start_monitoring()
    
    # 3. 기본 XGBoost 모델
    print("\n1. 기본 XGBoost 모델 훈련")
    xgb_basic = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_basic.fit(X_train, y_train)
    
    # 기본 모델 평가
    basic_results = evaluate_model_common(xgb_basic, X_train, X_test, y_train, y_test, "XGBoost_Basic")
    
    # 4. n_estimators 튜닝
    print("\n2. n_estimators 튜닝")
    n_estimators_range = [50, 100, 200, 300]
    train_scores, val_scores = validation_curve(
        XGBClassifier(random_state=42, eval_metric='mlogloss'), 
        X_train[:1500], y_train[:1500],  # 샘플 축소로 속도 향상
        param_name='n_estimators', param_range=n_estimators_range,
        cv=3, scoring='accuracy', n_jobs=-1
    )
    
    # 최적 n_estimators 찾기
    val_means = np.mean(val_scores, axis=1)
    best_n_estimators_idx = np.argmax(val_means)
    best_n_estimators = n_estimators_range[best_n_estimators_idx]
    
    print(f"최적 n_estimators: {best_n_estimators}")
    print(f"검증 정확도: {val_means[best_n_estimators_idx]:.4f}")
    
    # 학습 곡선 저장
    save_learning_curve(train_scores, val_scores, n_estimators_range, 'n_estimators', 'XGBoost')
    
    # 5. 그리드 서치로 하이퍼파라미터 최적화
    print("\n3. 하이퍼파라미터 최적화")
    param_grid = {
        'n_estimators': [best_n_estimators],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2]
    }
    
    grid_search = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='mlogloss'),
        param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    
    # 샘플 축소로 그리드 서치 속도 향상
    grid_search.fit(X_train[:1500], y_train[:1500])
    
    print("최적 파라미터:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"최적 교차검증 점수: {grid_search.best_score_:.4f}")
    
    # 6. 최적화된 모델 훈련
    print("\n4. 최적화된 XGBoost 모델 훈련")
    xgb_optimized = XGBClassifier(
        **grid_search.best_params_,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_optimized.fit(X_train, y_train)
    
    # 최적화 모델 평가
    optimized_results = evaluate_model_common(xgb_optimized, X_train, X_test, y_train, y_test, "XGBoost_Optimized")
    
    # 7. XGBoost 특화 분석
    print("\n5. XGBoost 특화 분석")
    
    # 특성 중요도 분석
    analyze_feature_importance(xgb_optimized)
    
    # 클래스별 분류 성능
    save_class_performance(optimized_results['classification_report'], "XGBoost")
    
    # 8. 성능 모니터링 종료 및 저장
    performance_data = monitor.end_monitoring()
    
    # 결과 데이터에 추가 정보 포함
    performance_data.update({
        'test_accuracy': optimized_results['test_accuracy'],
        'train_accuracy': optimized_results['train_accuracy'], 
        'prediction_time': optimized_results['prediction_time'],
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'n_features': X_train.shape[1]
    })
    
    save_performance_metrics(performance_data)
    
    print("\n6. XGBoost 분석 완료!")
    print(f"모든 결과는 results/ 폴더에 저장되었습니다.")
    
    return xgb_optimized, performance_data

def analyze_feature_importance(xgb_model):
    """특성 중요도 분석"""
    # 특성 중요도 가져오기
    importance_scores = xgb_model.feature_importances_
    
    # 상위 20개 중요 특성
    top_indices = np.argsort(importance_scores)[-20:]
    top_scores = importance_scores[top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), top_scores, color='darkgreen', alpha=0.7)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Index')
    plt.title('XGBoost - Top 20 Feature Importance')
    plt.yticks(range(20), [f'Pixel {i}' for i in top_indices])
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(IMAGES_DIR, 'XGBoost_feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"특성 중요도 저장: {save_path}")
    
    # 특성 중요도를 28x28 이미지로 시각화
    importance_image = importance_scores.reshape(28, 28)
    plt.figure(figsize=(8, 6))
    plt.imshow(importance_image, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('XGBoost - Feature Importance Heatmap (28x28)')
    
    save_path = os.path.join(IMAGES_DIR, 'XGBoost_importance_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"중요도 히트맵 저장: {save_path}")

def save_class_performance(classification_report, model_name):
    """클래스별 성능 시각화"""
    # F1-score 추출
    classes = [str(i) for i in range(10)]
    f1_scores = [classification_report[cls]['f1-score'] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1_scores, color='darkgreen', alpha=0.7)
    plt.xlabel('Class (Digit)')
    plt.ylabel('F1-Score')
    plt.title(f'{model_name} - Class-wise F1-Score')
    plt.ylim(0, 1)
    
    # 값 표시
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(IMAGES_DIR, f'{model_name}_class_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"클래스별 성능 저장: {save_path}")

if __name__ == "__main__":
    model, performance = xgboost_analysis() 