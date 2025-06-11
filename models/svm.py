"""
SVM을 이용한 MNIST 손글씨 분류
통일된 데이터셋과 성능 모니터링 적용
"""

from sklearn.svm import SVC
from sklearn.model_selection import validation_curve, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_common_mnist_data, preprocess_for_traditional_ml, 
    PerformanceMonitor, evaluate_model_common, save_performance_metrics,
    save_learning_curve, IMAGES_DIR
)
import os

def svm_analysis():
    """SVM 분석 실행"""
    print("=" * 60)
    print("SVM MNIST 분류 분석 (통일된 데이터셋)")
    print("=" * 60)
    
    # 1. 공통 데이터 로드
    X, y = load_common_mnist_data()
    X_train, X_test, y_train, y_test = preprocess_for_traditional_ml(X, y)
    
    # 2. 성능 모니터링 시작
    monitor = PerformanceMonitor("SVM")
    monitor.start_monitoring()
    
    # 3. 기본 SVM 모델 (Linear kernel)
    print("\n1. 기본 Linear SVM 모델 훈련")
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(X_train, y_train)
    
    # 기본 모델 평가
    linear_results = evaluate_model_common(svm_linear, X_train, X_test, y_train, y_test, "SVM_Linear")
    
    # 4. RBF 커널 SVM
    print("\n2. RBF 커널 SVM")
    svm_rbf = SVC(kernel='rbf', random_state=42)
    svm_rbf.fit(X_train, y_train)
    
    rbf_results = evaluate_model_common(svm_rbf, X_train, X_test, y_train, y_test, "SVM_RBF")
    
    # 5. 하이퍼파라미터 튜닝 (C 파라미터)
    print("\n3. C 파라미터 튜닝")
    C_range = [0.1, 1, 10, 100]
    train_scores, val_scores = validation_curve(
        SVC(kernel='rbf', random_state=42), X_train, y_train,
        param_name='C', param_range=C_range,
        cv=3, scoring='accuracy', n_jobs=-1
    )
    
    # 최적 C 값 찾기
    val_means = np.mean(val_scores, axis=1)
    best_C_idx = np.argmax(val_means)
    best_C = C_range[best_C_idx]
    
    print(f"최적 C 값: {best_C}")
    print(f"검증 정확도: {val_means[best_C_idx]:.4f}")
    
    # C 파라미터 학습 곡선 저장
    save_learning_curve(train_scores, val_scores, C_range, 'C', 'SVM')
    
    # 6. 감마 파라미터 튜닝
    print("\n4. 감마 파라미터 튜닝")
    gamma_range = [0.001, 0.01, 0.1, 1]
    train_scores_gamma, val_scores_gamma = validation_curve(
        SVC(kernel='rbf', C=best_C, random_state=42), X_train, y_train,
        param_name='gamma', param_range=gamma_range,
        cv=3, scoring='accuracy', n_jobs=-1
    )
    
    val_means_gamma = np.mean(val_scores_gamma, axis=1)
    best_gamma_idx = np.argmax(val_means_gamma)
    best_gamma = gamma_range[best_gamma_idx]
    
    print(f"최적 감마 값: {best_gamma}")
    print(f"검증 정확도: {val_means_gamma[best_gamma_idx]:.4f}")
    
    # 감마 파라미터 학습 곡선 저장
    save_learning_curve(train_scores_gamma, val_scores_gamma, gamma_range, 'gamma', 'SVM')
    
    # 7. 최적화된 SVM 모델
    print("\n5. 최적화된 SVM 모델 훈련")
    svm_optimized = SVC(
        kernel='rbf',
        C=best_C,
        gamma=best_gamma,
        random_state=42,
        probability=True  # 확률 추정 활성화
    )
    svm_optimized.fit(X_train, y_train)
    
    # 최적화 모델 평가
    optimized_results = evaluate_model_common(svm_optimized, X_train, X_test, y_train, y_test, "SVM_Optimized")
    
    # 8. SVM 특화 분석
    print("\n6. SVM 특화 분석")
    
    # 서포트 벡터 분석
    support_vector_analysis(svm_optimized, X_train, y_train)
    
    # 결정 함수 분석
    decision_function_analysis(svm_optimized, X_test, y_test)
    
    # 클래스별 분류 성능 시각화
    save_class_performance(optimized_results['classification_report'], "SVM")
    
    # 9. 성능 모니터링 종료 및 저장
    performance_data = monitor.end_monitoring()
    
    # 결과 데이터에 추가 정보 포함
    performance_data.update({
        'test_accuracy': optimized_results['test_accuracy'],
        'train_accuracy': optimized_results['train_accuracy'], 
        'prediction_time': optimized_results['prediction_time'],
        'best_C': best_C,
        'best_gamma': best_gamma,
        'n_support_vectors': np.sum(svm_optimized.n_support_),
        'kernel': 'rbf'
    })
    
    save_performance_metrics(performance_data)
    
    print("\n7. SVM 분석 완료!")
    print(f"모든 결과는 results/ 폴더에 저장되었습니다.")
    
    return svm_optimized, performance_data

def support_vector_analysis(svm_model, X_train, y_train):
    """서포트 벡터 분석"""
    print("\n서포트 벡터 분석:")
    print(f"총 서포트 벡터 수: {np.sum(svm_model.n_support_)}")
    print(f"전체 훈련 데이터 중 비율: {np.sum(svm_model.n_support_) / len(X_train) * 100:.2f}%")
    
    # 클래스별 서포트 벡터 수
    print("\n클래스별 서포트 벡터 수:")
    for i, n_sv in enumerate(svm_model.n_support_):
        print(f"클래스 {i}: {n_sv}개")
    
    # 클래스별 서포트 벡터 수 시각화
    plt.figure(figsize=(10, 6))
    classes = range(len(svm_model.n_support_))
    bars = plt.bar(classes, svm_model.n_support_, color='orange', alpha=0.7)
    plt.xlabel('Class (Digit)')
    plt.ylabel('Number of Support Vectors')
    plt.title('SVM - Support Vectors per Class')
    
    # 값 표시
    for bar, n_sv in zip(bars, svm_model.n_support_):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{n_sv}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(IMAGES_DIR, 'SVM_support_vectors.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"서포트 벡터 분석 저장: {save_path}")

def decision_function_analysis(svm_model, X_test, y_test):
    """결정 함수 분석"""
    # 결정 함수 값 계산
    decision_scores = svm_model.decision_function(X_test)
    
    # 각 클래스별 결정 함수 값 분포 시각화
    plt.figure(figsize=(15, 10))
    
    for i in range(10):
        plt.subplot(2, 5, i+1)
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            # 해당 클래스의 결정 함수 값들 중 해당 클래스에 대한 점수만 추출
            class_scores = decision_scores[class_mask]
            if decision_scores.ndim > 1:
                # 다중 클래스의 경우, 해당 클래스의 점수 찾기
                class_idx = list(svm_model.classes_).index(i)
                if decision_scores.shape[1] > class_idx:
                    scores_for_class = class_scores[:, class_idx] if class_scores.shape[1] > 1 else class_scores[:, 0]
                else:
                    scores_for_class = np.mean(class_scores, axis=1)
            else:
                scores_for_class = class_scores
            
            plt.hist(scores_for_class, bins=20, alpha=0.7, color=plt.cm.tab10(i))
            plt.title(f'Class {i}')
            plt.xlabel('Decision Score')
            plt.ylabel('Frequency')
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'SVM_decision_function.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"결정 함수 분석 저장: {save_path}")

def save_class_performance(classification_report, model_name):
    """클래스별 성능 시각화"""
    # F1-score 추출
    classes = [str(i) for i in range(10)]
    f1_scores = [classification_report[cls]['f1-score'] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1_scores, color='purple', alpha=0.7)
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
    model, performance = svm_analysis() 