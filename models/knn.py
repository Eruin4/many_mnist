"""
KNN을 이용한 MNIST 손글씨 분류
통일된 데이터셋과 성능 모니터링 적용
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_common_mnist_data, preprocess_for_traditional_ml, 
    PerformanceMonitor, evaluate_model_common, save_performance_metrics,
    save_learning_curve, IMAGES_DIR
)
import os

def knn_analysis():
    """KNN 분석 실행"""
    print("=" * 60)
    print("KNN MNIST 분류 분석 (통일된 데이터셋)")
    print("=" * 60)
    
    # 1. 공통 데이터 로드
    X, y = load_common_mnist_data()
    X_train, X_test, y_train, y_test = preprocess_for_traditional_ml(X, y)
    
    # 2. 성능 모니터링 시작
    monitor = PerformanceMonitor("KNN")
    monitor.start_monitoring()
    
    # 3. 기본 KNN 모델 (k=5)
    print("\n1. 기본 KNN 모델 훈련 (k=5)")
    knn_basic = KNeighborsClassifier(n_neighbors=5)
    knn_basic.fit(X_train, y_train)
    
    # 기본 모델 평가
    basic_results = evaluate_model_common(knn_basic, X_train, X_test, y_train, y_test, "KNN_Basic")
    
    # 4. 최적 K 값 찾기
    print("\n2. 최적 K 값 찾기")
    k_range = [1, 3, 5, 7, 9, 11, 15, 20]
    train_scores, val_scores = validation_curve(
        KNeighborsClassifier(), X_train, y_train,
        param_name='n_neighbors', param_range=k_range,
        cv=3, scoring='accuracy', n_jobs=-1
    )
    
    # 최적 K 값 찾기
    val_means = np.mean(val_scores, axis=1)
    best_k_idx = np.argmax(val_means)
    best_k = k_range[best_k_idx]
    
    print(f"최적 K 값: {best_k}")
    print(f"검증 정확도: {val_means[best_k_idx]:.4f}")
    
    # K 값 학습 곡선 저장
    save_learning_curve(train_scores, val_scores, k_range, 'n_neighbors', 'KNN')
    
    # 5. 거리 메트릭 비교
    print("\n3. 다양한 거리 메트릭 비교")
    metrics = ['euclidean', 'manhattan', 'chebyshev']
    metric_results = {}
    
    for metric in metrics:
        print(f"{metric} 메트릭 테스트중...")
        try:
            knn_metric = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
            knn_metric.fit(X_train, y_train)
            accuracy = knn_metric.score(X_test, y_test)
            metric_results[metric] = accuracy
            print(f"{metric}: {accuracy:.4f}")
        except Exception as e:
            print(f"{metric} 오류: {e}")
            metric_results[metric] = 0
    
    # 최적 메트릭 선택
    best_metric = max(metric_results, key=metric_results.get)
    print(f"최적 거리 메트릭: {best_metric}")
    
    # 6. 가중치 방법 비교
    print("\n4. 가중치 방법 비교")
    weights = ['uniform', 'distance']
    weight_results = {}
    
    for weight in weights:
        print(f"{weight} 가중치 테스트중...")
        knn_weight = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights=weight)
        knn_weight.fit(X_train, y_train)
        accuracy = knn_weight.score(X_test, y_test)
        weight_results[weight] = accuracy
        print(f"{weight}: {accuracy:.4f}")
    
    # 최적 가중치 선택
    best_weight = max(weight_results, key=weight_results.get)
    print(f"최적 가중치: {best_weight}")
    
    # 7. 최적화된 모델 훈련
    print("\n5. 최적화된 KNN 모델 훈련")
    knn_optimized = KNeighborsClassifier(
        n_neighbors=best_k,
        metric=best_metric,
        weights=best_weight
    )
    knn_optimized.fit(X_train, y_train)
    
    # 최적화 모델 평가
    optimized_results = evaluate_model_common(knn_optimized, X_train, X_test, y_train, y_test, "KNN_Optimized")
    
    # 8. KNN 특화 분석
    print("\n6. KNN 특화 분석")
    
    # 이웃 분석
    analyze_neighbors(knn_optimized, X_train, X_test, y_train, y_test)
    
    # 거리 분포 분석
    analyze_distance_distribution(knn_optimized, X_train, X_test)
    
    # 클래스별 분류 성능 시각화
    save_class_performance(optimized_results['classification_report'], "KNN")
    
    # 9. 성능 모니터링 종료 및 저장
    performance_data = monitor.end_monitoring()
    
    # 결과 데이터에 추가 정보 포함
    performance_data.update({
        'test_accuracy': optimized_results['test_accuracy'],
        'train_accuracy': optimized_results['train_accuracy'], 
        'prediction_time': optimized_results['prediction_time'],
        'best_k': best_k,
        'best_metric': best_metric,
        'best_weight': best_weight,
        'training_set_size': len(X_train)
    })
    
    save_performance_metrics(performance_data)
    
    print("\n7. KNN 분석 완료!")
    print(f"모든 결과는 results/ 폴더에 저장되었습니다.")
    
    return knn_optimized, performance_data

def analyze_neighbors(knn_model, X_train, X_test, y_train, y_test):
    """이웃 분석"""
    # 테스트 샘플의 이웃들 분석
    n_samples = 5  # 분석할 테스트 샘플 수
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, knn_model.n_neighbors + 1, figsize=(15, n_samples * 2))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample_idx in enumerate(sample_indices):
        # 테스트 샘플
        test_sample = X_test[sample_idx:sample_idx+1]
        true_label = y_test[sample_idx]
        
        # 이웃 찾기
        distances, indices = knn_model.kneighbors(test_sample)
        neighbor_labels = y_train[indices[0]]
        
        # 테스트 샘플 시각화
        axes[i, 0].imshow(test_sample.reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f'Test\nTrue: {true_label}')
        axes[i, 0].axis('off')
        
        # 이웃들 시각화
        for j in range(knn_model.n_neighbors):
            neighbor_idx = indices[0][j]
            neighbor_data = X_train[neighbor_idx].reshape(28, 28)
            neighbor_label = neighbor_labels[j]
            distance = distances[0][j]
            
            axes[i, j+1].imshow(neighbor_data, cmap='gray')
            axes[i, j+1].set_title(f'Neighbor {j+1}\nLabel: {neighbor_label}\nDist: {distance:.2f}')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'KNN_neighbor_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"이웃 분석 저장: {save_path}")

def analyze_distance_distribution(knn_model, X_train, X_test):
    """거리 분포 분석"""
    # 랜덤 테스트 샘플들의 거리 분포 분석
    n_samples = 100
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    all_distances = []
    
    for sample_idx in sample_indices:
        test_sample = X_test[sample_idx:sample_idx+1]
        distances, indices = knn_model.kneighbors(test_sample)
        all_distances.extend(distances[0])
    
    plt.figure(figsize=(10, 6))
    
    # 거리 분포
    plt.hist(all_distances, bins=30, alpha=0.7, color='skyblue')
    plt.xlabel('Distance to Neighbors')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances to Neighbors')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'KNN_distance_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"거리 분포 분석 저장: {save_path}")

def save_class_performance(classification_report, model_name):
    """클래스별 성능 시각화"""
    # F1-score 추출
    classes = [str(i) for i in range(10)]
    f1_scores = [classification_report[cls]['f1-score'] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1_scores, color='lightblue', alpha=0.7)
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
    model, performance = knn_analysis() 