"""
CNN을 이용한 MNIST 손글씨 분류
통일된 데이터셋과 성능 모니터링 적용
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os
from utils import (
    load_common_mnist_data, PerformanceMonitor, 
    save_confusion_matrix, save_classification_report,
    save_performance_metrics, IMAGES_DIR
)

def cnn_analysis():
    """CNN 분석 실행"""
    print("=" * 60)
    print("CNN MNIST 분류 분석 (통일된 데이터셋)")
    print("=" * 60)
    
    # 1. 공통 데이터 로드
    X, y = load_common_mnist_data()
    
    # CNN용 데이터 전처리
    X_train, X_test, y_train, y_test = preprocess_for_cnn(X, y)
    
    print(f"훈련 데이터 형태: {X_train.shape}")
    print(f"테스트 데이터 형태: {X_test.shape}")
    
    # 2. 성능 모니터링 시작
    monitor = PerformanceMonitor("CNN")
    monitor.start_monitoring()
    
    # 3. 기본 CNN 모델
    print("\n1. 기본 CNN 모델 구축")
    basic_model = create_basic_cnn()
    basic_model.summary()
    
    print("\n2. 기본 CNN 모델 훈련")
    basic_history = basic_model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # 기본 모델 평가
    basic_results = evaluate_cnn_model(basic_model, X_train, X_test, y_train, y_test, "CNN_Basic")
    
    # 4. 개선된 CNN 모델
    print("\n3. 개선된 CNN 모델 구축")
    improved_model = create_improved_cnn()
    improved_model.summary()
    
    print("\n4. 개선된 CNN 모델 훈련")
    improved_history = improved_model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # 개선된 모델 평가
    improved_results = evaluate_cnn_model(improved_model, X_train, X_test, y_train, y_test, "CNN_Improved")
    
    # 5. 훈련 과정 시각화
    save_training_history(basic_history, "CNN_Basic")
    save_training_history(improved_history, "CNN_Improved")
    
    # 6. CNN 특화 분석
    print("\n5. CNN 특화 분석")
    
    # 모델 비교
    compare_models(basic_results, improved_results)
    
    # 예측 시각화
    visualize_predictions(improved_model, X_test, y_test)
    
    # 클래스별 성능
    save_class_performance(improved_results['classification_report'], "CNN")
    
    # 7. 성능 모니터링 종료
    performance_data = monitor.end_monitoring()
    
    # 최고 성능 모델 선택
    best_model = improved_model if improved_results['test_accuracy'] > basic_results['test_accuracy'] else basic_model
    best_results = improved_results if improved_results['test_accuracy'] > basic_results['test_accuracy'] else basic_results
    
    performance_data.update({
        'test_accuracy': best_results['test_accuracy'],
        'train_accuracy': best_results['train_accuracy'],
        'prediction_time': best_results['prediction_time'],
        'basic_accuracy': basic_results['test_accuracy'],
        'improved_accuracy': improved_results['test_accuracy'],
        'model_type': 'CNN'
    })
    
    save_performance_metrics(performance_data)
    
    print(f"\n6. CNN 분석 완료!")
    print(f"기본 모델 정확도: {basic_results['test_accuracy']:.4f}")
    print(f"개선 모델 정확도: {improved_results['test_accuracy']:.4f}")
    print(f"모든 결과는 results/ 폴더에 저장되었습니다.")
    
    return best_model, performance_data

def preprocess_for_cnn(X, y):
    """CNN용 데이터 전처리"""
    from sklearn.model_selection import train_test_split
    
    # 훈련/테스트 분할 (통일된 방식)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 이미지 형태로 변환 (28x28x1)
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # 원-핫 인코딩
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return X_train, X_test, y_train, y_test

def create_basic_cnn():
    """기본 CNN 모델 생성"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_improved_cnn():
    """개선된 CNN 모델 생성"""
    model = keras.Sequential([
        # 첫 번째 블록
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 두 번째 블록
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 세 번째 블록
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.25),
        
        # 완전연결층
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def evaluate_cnn_model(model, X_train, X_test, y_train, y_test, model_name):
    """CNN 모델 평가"""
    print(f"{model_name} 모델 평가중...")
    
    # 예측 시간 측정
    import time
    start_time = time.time()
    
    train_pred_proba = model.predict(X_train, verbose=0)
    test_pred_proba = model.predict(X_test, verbose=0)
    
    prediction_time = time.time() - start_time
    
    # 클래스 예측 (원-핫에서 클래스로 변환)
    train_pred = np.argmax(train_pred_proba, axis=1)
    test_pred = np.argmax(test_pred_proba, axis=1)
    
    y_train_classes = np.argmax(y_train, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # 정확도 계산
    train_accuracy = accuracy_score(y_train_classes, train_pred)
    test_accuracy = accuracy_score(y_test_classes, test_pred)
    
    print(f"훈련 정확도: {train_accuracy:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"예측 시간: {prediction_time:.4f}초")
    
    # 결과 저장
    save_confusion_matrix(y_test_classes, test_pred, model_name)
    classification_report_data = save_classification_report(y_test_classes, test_pred, model_name)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'prediction_time': prediction_time,
        'test_pred': test_pred,
        'test_proba': test_pred_proba,
        'classification_report': classification_report_data
    }

def save_training_history(history, model_name):
    """훈련 과정 시각화 및 저장"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 정확도 그래프
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title(f'{model_name} - Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 손실 그래프
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title(f'{model_name} - Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"훈련 과정 저장: {save_path}")

def compare_models(basic_results, improved_results):
    """기본 모델과 개선 모델 비교"""
    models = ['Basic CNN', 'Improved CNN']
    accuracies = [basic_results['test_accuracy'], improved_results['test_accuracy']]
    pred_times = [basic_results['prediction_time'], improved_results['prediction_time']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 정확도 비교
    bars1 = ax1.bar(models, accuracies, color=['lightblue', 'darkblue'], alpha=0.7)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('CNN Models - Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom')
    
    # 예측 시간 비교
    bars2 = ax2.bar(models, pred_times, color=['lightcoral', 'darkred'], alpha=0.7)
    ax2.set_ylabel('Prediction Time (seconds)')
    ax2.set_title('CNN Models - Speed Comparison')
    
    for bar, time in zip(bars2, pred_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pred_times)*0.02,
                f'{time:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'CNN_model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"모델 비교 저장: {save_path}")

def visualize_predictions(model, X_test, y_test, n_samples=16):
    """예측 결과 시각화"""
    # 랜덤 샘플 선택
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    predictions = model.predict(X_test[indices], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[indices], axis=1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(n_samples):
        ax = axes[i//4, i%4]
        
        # 이미지 표시
        ax.imshow(X_test[indices[i]].reshape(28, 28), cmap='gray')
        
        # 제목 설정 (실제 vs 예측)
        true_label = true_classes[i]
        pred_label = predicted_classes[i]
        confidence = predictions[i][pred_label]
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('CNN - Prediction Examples', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, 'CNN_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"예측 예시 저장: {save_path}")

def save_class_performance(classification_report, model_name):
    """클래스별 성능 시각화"""
    # F1-score 추출
    classes = [str(i) for i in range(10)]
    f1_scores = [classification_report[cls]['f1-score'] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1_scores, color='darkblue', alpha=0.7)
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
    model, performance = cnn_analysis() 