"""
랜덤포레스트를 이용한 MNIST 손글씨 분류
통일된 데이터셋과 성능 모니터링 적용
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_common_mnist_data, preprocess_for_traditional_ml, 
    PerformanceMonitor, evaluate_model_common, save_performance_metrics,
    save_feature_importance, save_learning_curve, IMAGES_DIR
)
import os

def random_forest_analysis():
    """랜덤포레스트 분석 실행"""
    print("=" * 60)
    print("랜덤포레스트 MNIST 분류 분석 (통일된 데이터셋)")
    print("=" * 60)
    
    # 1. 공통 데이터 로드
    X, y = load_common_mnist_data()
    X_train, X_test, y_train, y_test = preprocess_for_traditional_ml(X, y)
    
    # 2. 성능 모니터링 시작
    monitor = PerformanceMonitor("RandomForest")
    monitor.start_monitoring()
    
    # 3. 기본 모델 훈련
    print("\n1. 기본 랜덤포레스트 모델 훈련")
    rf_basic = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_basic.fit(X_train, y_train)
    
    # 기본 모델 평가
    basic_results = evaluate_model_common(rf_basic, X_train, X_test, y_train, y_test, "RandomForest_Basic")
    
    # 4. 하이퍼파라미터 튜닝 - n_estimators
    print("\n2. 최적 트리 수 찾기")
    n_estimators_range = [10, 50, 100, 200, 300]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42, n_jobs=-1), X_train, y_train,
        param_name='n_estimators', param_range=n_estimators_range,
        cv=3, scoring='accuracy', n_jobs=-1
    )
    
    # 최적 트리 수 찾기
    val_means = np.mean(val_scores, axis=1)
    best_n_estimators_idx = np.argmax(val_means)
    best_n_estimators = n_estimators_range[best_n_estimators_idx]
    
    print(f"최적 트리 수: {best_n_estimators}")
    print(f"검증 정확도: {val_means[best_n_estimators_idx]:.4f}")
    
    # 학습 곡선 저장
    save_learning_curve(train_scores, val_scores, n_estimators_range, 
                        'n_estimators', 'RandomForest')
    
    # 5. max_depth 튜닝
    print("\n3. 최적 트리 깊이 찾기")
    max_depths = [5, 10, 15, 20, None]
    train_scores_depth, val_scores_depth = validation_curve(
        RandomForestClassifier(n_estimators=best_n_estimators, random_state=42, n_jobs=-1), 
        X_train, y_train,
        param_name='max_depth', param_range=max_depths,
        cv=3, scoring='accuracy', n_jobs=-1
    )
    
    val_means_depth = np.mean(val_scores_depth, axis=1)
    best_depth_idx = np.argmax(val_means_depth)
    best_depth = max_depths[best_depth_idx]
    
    print(f"최적 트리 깊이: {best_depth}")
    print(f"검증 정확도: {val_means_depth[best_depth_idx]:.4f}")
    
    # 6. 최적화된 모델 훈련
    print("\n4. 최적화된 모델 훈련")
    rf_optimized = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        oob_score=True  # Out-of-bag 점수 계산
    )
    rf_optimized.fit(X_train, y_train)
    
    # 최적화 모델 평가
    optimized_results = evaluate_model_common(rf_optimized, X_train, X_test, y_train, y_test, "RandomForest_Optimized")
    
    # 7. 랜덤포레스트 특화 분석
    print("\n5. 랜덤포레스트 특화 분석")
    
    # OOB 점수
    oob_score = rf_optimized.oob_score_
    print(f"Out-of-bag 점수: {oob_score:.4f}")
    
    # 트리별 통계
    forest_stats = analyze_forest_structure(rf_optimized)
    print(f"평균 트리 깊이: {forest_stats['avg_depth']:.2f}")
    print(f"평균 잎 노드 수: {forest_stats['avg_leaves']:.2f}")
    print(f"평균 노드 수: {forest_stats['avg_nodes']:.2f}")
    
    # 8. 특성 중요도 분석 및 저장
    print("\n6. 특성 중요도 분석")
    save_feature_importance(rf_optimized.feature_importances_, "RandomForest")
    
    # 트리별 중요도 분산 시각화
    save_feature_importance_variance(rf_optimized)
    
    # 클래스별 분류 성능 시각화
    save_class_performance(optimized_results['classification_report'], "RandomForest")
    
    # 9. 성능 모니터링 종료 및 저장
    performance_data = monitor.end_monitoring()
    
    # 결과 데이터에 추가 정보 포함
    performance_data.update({
        'test_accuracy': optimized_results['test_accuracy'],
        'train_accuracy': optimized_results['train_accuracy'], 
        'prediction_time': optimized_results['prediction_time'],
        'n_estimators': best_n_estimators,
        'max_depth': best_depth,
        'oob_score': oob_score,
        'avg_tree_depth': forest_stats['avg_depth'],
        'avg_leaves': forest_stats['avg_leaves']
    })
    
    save_performance_metrics(performance_data)
    
    print("\n7. 랜덤포레스트 분석 완료!")
    print(f"모든 결과는 results/ 폴더에 저장되었습니다.")
    
    return rf_optimized, performance_data

def analyze_forest_structure(rf_model):
    """포레스트 구조 분석"""
    depths = []
    leaves = []
    nodes = []
    
    for tree in rf_model.estimators_:
        depths.append(tree.tree_.max_depth)
        leaves.append(tree.tree_.n_leaves)
        nodes.append(tree.tree_.node_count)
    
    return {
        'avg_depth': np.mean(depths),
        'avg_leaves': np.mean(leaves),
        'avg_nodes': np.mean(nodes),
        'depths': depths,
        'leaves': leaves,
        'nodes': nodes
    }

def save_feature_importance_variance(rf_model):
    """트리별 특성 중요도 분산 시각화"""
    # 각 트리의 특성 중요도 수집
    importances = np.array([tree.feature_importances_ for tree in rf_model.estimators_])
    
    # 평균과 표준편차 계산
    mean_importance = np.mean(importances, axis=0)
    std_importance = np.std(importances, axis=0)
    
    # 상위 20개 특성 선택
    top_indices = np.argsort(mean_importance)[-20:]
    
    plt.figure(figsize=(12, 8))
    plt.errorbar(range(20), mean_importance[top_indices], 
                yerr=std_importance[top_indices], fmt='o', capsize=5)
    plt.xlabel('Feature Index (Top 20)')
    plt.ylabel('Importance')
    plt.title('RandomForest - Feature Importance with Variance')
    plt.xticks(range(20), [f'Pixel_{i}' for i in top_indices], rotation=45)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(IMAGES_DIR, 'RandomForest_importance_variance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"특성 중요도 분산 저장: {save_path}")

def save_class_performance(classification_report, model_name):
    """클래스별 성능 시각화"""
    # F1-score 추출
    classes = [str(i) for i in range(10)]
    f1_scores = [classification_report[cls]['f1-score'] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1_scores, color='forestgreen', alpha=0.7)
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
    model, performance = random_forest_analysis() 