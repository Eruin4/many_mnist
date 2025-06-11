"""
로지스틱 회귀를 이용한 MNIST 손글씨 분류
통일된 데이터셋과 성능 모니터링 적용
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_common_mnist_data, preprocess_for_traditional_ml, 
    PerformanceMonitor, evaluate_model_common, save_performance_metrics,
    save_learning_curve, IMAGES_DIR
)
import os

def logistic_regression_analysis():
    """로지스틱 회귀 분석 실행"""
    print("=" * 60)
    print("로지스틱 회귀 MNIST 분류 분석 (통일된 데이터셋)")
    print("=" * 60)
    
    # 1. 공통 데이터 로드
    X, y = load_common_mnist_data()
    X_train, X_test, y_train, y_test = preprocess_for_traditional_ml(X, y)
    
    # 2. 성능 모니터링 시작
    monitor = PerformanceMonitor("LogisticRegression")
    monitor.start_monitoring()
    
    # 3. 기본 로지스틱 회귀 모델
    print("\n1. 기본 로지스틱 회귀 모델 훈련")
    lr_basic = LogisticRegression(random_state=42, max_iter=1000)
    lr_basic.fit(X_train, y_train)
    
    # 기본 모델 평가
    basic_results = evaluate_model_common(lr_basic, X_train, X_test, y_train, y_test, "LogisticRegression_Basic")
    
    # 4. 정규화 파라미터 튜닝 (C 파라미터)
    print("\n2. 정규화 강도 튜닝 (C 파라미터)")
    C_range = [0.01, 0.1, 1, 10, 100]
    train_scores, val_scores = validation_curve(
        LogisticRegression(random_state=42, max_iter=1000), X_train, y_train,
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
    save_learning_curve(train_scores, val_scores, C_range, 'C', 'LogisticRegression')
    
    # 5. 솔버 비교
    print("\n3. 다양한 솔버 비교")
    solvers = ['liblinear', 'lbfgs', 'saga']
    solver_results = {}
    
    for solver in solvers:
        print(f"{solver} 솔버 테스트중...")
        try:
            lr_solver = LogisticRegression(C=best_C, solver=solver, random_state=42, max_iter=1000)
            lr_solver.fit(X_train, y_train)
            accuracy = lr_solver.score(X_test, y_test)
            solver_results[solver] = accuracy
            print(f"{solver}: {accuracy:.4f}")
        except Exception as e:
            print(f"{solver} 오류: {e}")
            solver_results[solver] = 0
    
    # 최적 솔버 선택
    best_solver = max(solver_results, key=solver_results.get)
    print(f"최적 솔버: {best_solver}")
    
    # 6. 최적화된 모델 훈련
    print("\n4. 최적화된 로지스틱 회귀 모델 훈련")
    lr_optimized = LogisticRegression(
        C=best_C,
        solver=best_solver,
        random_state=42,
        max_iter=2000
    )
    lr_optimized.fit(X_train, y_train)
    
    # 최적화 모델 평가
    optimized_results = evaluate_model_common(lr_optimized, X_train, X_test, y_train, y_test, "LogisticRegression_Optimized")
    
    # 7. 로지스틱 회귀 특화 분석
    print("\n5. 로지스틱 회귀 특화 분석")
    
    # 계수 분석
    analyze_coefficients(lr_optimized)
    
    # 예측 확률 분석
    analyze_prediction_probabilities(lr_optimized, X_test, y_test)
    
    # 클래스별 분류 성능 시각화
    save_class_performance(optimized_results['classification_report'], "LogisticRegression")
    
    # 8. 성능 모니터링 종료 및 저장
    performance_data = monitor.end_monitoring()
    
    # 결과 데이터에 추가 정보 포함
    performance_data.update({
        'test_accuracy': optimized_results['test_accuracy'],
        'train_accuracy': optimized_results['train_accuracy'], 
        'prediction_time': optimized_results['prediction_time'],
        'best_C': best_C,
        'best_solver': best_solver,
        'n_features': X_train.shape[1],
        'n_classes': len(np.unique(y_train))
    })
    
    save_performance_metrics(performance_data)
    
    print("\n6. 로지스틱 회귀 분석 완료!")
    print(f"모든 결과는 results/ 폴더에 저장되었습니다.")
    
    return lr_optimized, performance_data

def analyze_coefficients(lr_model):
    """계수 분석 및 시각화"""
    coefficients = lr_model.coef_
    
    # 각 클래스별 계수의 크기 분석
    coef_norms = np.linalg.norm(coefficients, axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(10), coef_norms, color='orange', alpha=0.7)
    plt.xlabel('Class (Digit)')
    plt.ylabel('Coefficient Norm')
    plt.title('LogisticRegression - Coefficient Norms per Class')
    plt.xticks(range(10))
    
    # 값 표시
    for bar, norm in zip(bars, coef_norms):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(coef_norms)*0.01,
                f'{norm:.2f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(IMAGES_DIR, 'LogisticRegression_coefficient_norms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"계수 분석 저장: {save_path}")
    
    # 첫 번째 클래스의 계수를 28x28 이미지로 시각화
    coef_image = coefficients[0].reshape(28, 28)
    plt.figure(figsize=(8, 6))
    plt.imshow(coef_image, cmap='RdBu', interpolation='nearest')
    plt.colorbar()
    plt.title('LogisticRegression - Coefficients for Class 0 (as 28x28 image)')
    
    save_path = os.path.join(IMAGES_DIR, 'LogisticRegression_coefficients_class0.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"계수 시각화 저장: {save_path}")

def analyze_prediction_probabilities(lr_model, X_test, y_test):
    """예측 확률 분석"""
    probabilities = lr_model.predict_proba(X_test)
    predictions = lr_model.predict(X_test)
    
    # 정확한 예측과 틀린 예측의 확률 분포 비교
    correct_mask = (predictions == y_test)
    correct_probs = np.max(probabilities[correct_mask], axis=1)
    incorrect_probs = np.max(probabilities[~correct_mask], axis=1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(correct_probs, bins=20, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(incorrect_probs, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.xlabel('Maximum Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 클래스별 평균 확률
    plt.subplot(1, 2, 2)
    class_avg_probs = []
    for i in range(10):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            avg_prob = np.mean(np.max(probabilities[class_mask], axis=1))
            class_avg_probs.append(avg_prob)
        else:
            class_avg_probs.append(0)
    
    bars = plt.bar(range(10), class_avg_probs, color='skyblue', alpha=0.7)
    plt.xlabel('Class (Digit)')
    plt.ylabel('Average Max Probability')
    plt.title('Average Prediction Confidence per Class')
    plt.xticks(range(10))
    
    for bar, prob in zip(bars, class_avg_probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(IMAGES_DIR, 'LogisticRegression_probability_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"확률 분석 저장: {save_path}")

def save_class_performance(classification_report, model_name):
    """클래스별 성능 시각화"""
    # F1-score 추출
    classes = [str(i) for i in range(10)]
    f1_scores = [classification_report[cls]['f1-score'] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1_scores, color='coral', alpha=0.7)
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
    model, performance = logistic_regression_analysis() 