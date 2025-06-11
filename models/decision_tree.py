"""
의사결정나무를 이용한 MNIST 손글씨 분류
통일된 데이터셋과 성능 모니터링 적용
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_common_mnist_data, preprocess_for_traditional_ml, 
    PerformanceMonitor, evaluate_model_common, save_performance_metrics,
    save_feature_importance, save_learning_curve, IMAGES_DIR
)
import os

def decision_tree_analysis():
    """의사결정나무 분석 실행"""
    print("=" * 60)
    print("의사결정나무 MNIST 분류 분석 (통일된 데이터셋)")
    print("=" * 60)
    
    # 1. 공통 데이터 로드
    X, y = load_common_mnist_data()
    X_train, X_test, y_train, y_test = preprocess_for_traditional_ml(X, y)
    
    # 2. 성능 모니터링 시작
    monitor = PerformanceMonitor("DecisionTree")
    monitor.start_monitoring()
    
    # 3. 기본 모델 훈련
    print("\n1. 기본 의사결정나무 모델 훈련")
    dt_basic = DecisionTreeClassifier(random_state=42)
    dt_basic.fit(X_train, y_train)
    
    # 기본 모델 평가
    basic_results = evaluate_model_common(dt_basic, X_train, X_test, y_train, y_test, "DecisionTree_Basic")
    
    # 4. 하이퍼파라미터 튜닝
    print("\n2. 최적 트리 깊이 찾기")
    max_depths = [3, 5, 7, 10, 15, 20, None]
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(random_state=42), X_train, y_train,
        param_name='max_depth', param_range=max_depths,
        cv=3, scoring='accuracy', n_jobs=-1
    )
    
    # 최적 깊이 찾기
    val_means = np.mean(val_scores, axis=1)
    best_depth_idx = np.argmax(val_means)
    best_depth = max_depths[best_depth_idx]
    
    print(f"최적 트리 깊이: {best_depth}")
    print(f"검증 정확도: {val_means[best_depth_idx]:.4f}")
    
    # 학습 곡선 저장
    depth_range_for_plot = [3, 5, 7, 10, 15, 20, 25]
    save_learning_curve(train_scores[:6], val_scores[:6], depth_range_for_plot[:6], 
                        'max_depth', 'DecisionTree')
    
    # 5. 최적화된 모델 훈련
    print("\n3. 최적화된 모델 훈련")
    dt_optimized = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt_optimized.fit(X_train, y_train)
    
    # 최적화 모델 평가
    optimized_results = evaluate_model_common(dt_optimized, X_train, X_test, y_train, y_test, "DecisionTree_Optimized")
    
    # 6. 트리 통계 분석
    print("\n4. 트리 구조 분석")
    tree_stats = {
        'tree_depth': dt_optimized.tree_.max_depth,
        'leaf_count': dt_optimized.tree_.n_leaves,
        'node_count': dt_optimized.tree_.node_count,
        'feature_count': dt_optimized.tree_.n_features
    }
    
    print(f"트리 깊이: {tree_stats['tree_depth']}")
    print(f"잎 노드 수: {tree_stats['leaf_count']}")
    print(f"전체 노드 수: {tree_stats['node_count']}")
    print(f"사용된 특성 수: {tree_stats['feature_count']}")
    
    # 7. 특성 중요도 분석 및 저장
    print("\n5. 특성 중요도 분석")
    save_feature_importance(dt_optimized.feature_importances_, "DecisionTree")
    
    # 클래스별 분류 성능 시각화
    save_class_performance(optimized_results['classification_report'], "DecisionTree")
    
    # 8. 성능 모니터링 종료 및 저장
    performance_data = monitor.end_monitoring()
    
    # 결과 데이터에 추가 정보 포함
    performance_data.update({
        'test_accuracy': optimized_results['test_accuracy'],
        'train_accuracy': optimized_results['train_accuracy'], 
        'prediction_time': optimized_results['prediction_time'],
        'tree_depth': tree_stats['tree_depth'],
        'leaf_count': tree_stats['leaf_count'],
        'node_count': tree_stats['node_count']
    })
    
    save_performance_metrics(performance_data)
    
    print("\n6. 의사결정나무 분석 완료!")
    print(f"모든 결과는 results/ 폴더에 저장되었습니다.")
    
    return dt_optimized, performance_data

def save_class_performance(classification_report, model_name):
    """클래스별 성능 시각화"""
    # F1-score 추출
    classes = [str(i) for i in range(10)]
    f1_scores = [classification_report[cls]['f1-score'] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1_scores, color='skyblue', alpha=0.7)
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
    model, performance = decision_tree_analysis() 