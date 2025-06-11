"""
모든 모델의 종합 성능 비교 차트 생성
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager, rc
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_performance_data():
    """성능 데이터 로드"""
    with open('results/performance_metrics.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_performance_comparison():
    """종합 성능 비교 차트 생성"""
    data = load_performance_data()
    
    # 모델 이름과 메트릭 추출
    models = []
    test_accuracies = []
    train_accuracies = []
    training_times = []
    prediction_times = []
    memory_usage = []
    
    for model_name, metrics in data.items():
        models.append(model_name)
        test_accuracies.append(metrics['test_accuracy'] * 100)  # 백분율로 변환
        train_accuracies.append(metrics['train_accuracy'] * 100)
        training_times.append(metrics['training_time'])
        prediction_times.append(metrics['prediction_time'])
        memory_usage.append(abs(metrics['memory_used']))  # 절댓값 사용
    
    # 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MNIST 모델 종합 성능 비교', fontsize=20, fontweight='bold')
    
    # 1. 정확도 비교 (막대 그래프)
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_accuracies, width, label='훈련 정확도', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_accuracies, width, label='테스트 정확도', color='orange', alpha=0.8)
    
    ax1.set_xlabel('모델')
    ax1.set_ylabel('정확도 (%)')
    ax1.set_title('훈련 vs 테스트 정확도 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 막대 위에 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. 테스트 정확도 순위 (수평 막대 그래프)
    ax2 = axes[0, 1]
    sorted_indices = np.argsort(test_accuracies)[::-1]  # 내림차순 정렬
    sorted_models = [models[i] for i in sorted_indices]
    sorted_accuracies = [test_accuracies[i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_models)))
    bars = ax2.barh(sorted_models, sorted_accuracies, color=colors, alpha=0.8)
    
    ax2.set_xlabel('테스트 정확도 (%)')
    ax2.set_title('테스트 정확도 순위')
    ax2.grid(True, alpha=0.3)
    
    # 막대 끝에 값 표시
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}%', ha='left', va='center', fontweight='bold')
    
    # 3. 훈련 시간 비교 (로그 스케일)
    ax3 = axes[0, 2]
    bars = ax3.bar(models, training_times, color='lightcoral', alpha=0.8)
    ax3.set_xlabel('모델')
    ax3.set_ylabel('훈련 시간 (초, 로그 스케일)')
    ax3.set_title('훈련 시간 비교')
    ax3.set_yscale('log')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # 4. 예측 시간 비교 (로그 스케일)
    ax4 = axes[1, 0]
    bars = ax4.bar(models, prediction_times, color='lightgreen', alpha=0.8)
    ax4.set_xlabel('모델')
    ax4.set_ylabel('예측 시간 (초, 로그 스케일)')
    ax4.set_title('예측 시간 비교')
    ax4.set_yscale('log')
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # 5. 메모리 사용량 비교
    ax5 = axes[1, 1]
    bars = ax5.bar(models, memory_usage, color='gold', alpha=0.8)
    ax5.set_xlabel('모델')
    ax5.set_ylabel('메모리 사용량 (MB)')
    ax5.set_title('메모리 사용량 비교')
    ax5.set_xticklabels(models, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(memory_usage)*0.01,
                f'{height:.1f}MB', ha='center', va='bottom', fontsize=8)
    
    # 6. 성능 vs 효율성 산점도
    ax6 = axes[1, 2]
    scatter = ax6.scatter(training_times, test_accuracies, 
                         s=[m*2 for m in memory_usage], 
                         c=range(len(models)), 
                         cmap='tab10', alpha=0.7)
    
    # 모델 이름 레이블 추가
    for i, model in enumerate(models):
        ax6.annotate(model, (training_times[i], test_accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    ax6.set_xlabel('훈련 시간 (초)')
    ax6.set_ylabel('테스트 정확도 (%)')
    ax6.set_title('성능 vs 훈련시간 (원 크기: 메모리 사용량)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 이미지 저장
    os.makedirs('results/images/Overall_Analysis', exist_ok=True)
    plt.savefig('results/images/Overall_Analysis/모델_종합_성능_비교.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def create_overfitting_analysis():
    """과적합 분석 차트"""
    data = load_performance_data()
    
    models = []
    overfitting_scores = []
    test_accuracies = []
    
    for model_name, metrics in data.items():
        models.append(model_name)
        overfitting = (metrics['train_accuracy'] - metrics['test_accuracy']) * 100
        overfitting_scores.append(overfitting)
        test_accuracies.append(metrics['test_accuracy'] * 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('과적합 분석', fontsize=16, fontweight='bold')
    
    # 1. 과적합 정도 비교
    colors = ['red' if score > 10 else 'orange' if score > 5 else 'green' 
              for score in overfitting_scores]
    bars = ax1.bar(models, overfitting_scores, color=colors, alpha=0.7)
    
    ax1.set_xlabel('모델')
    ax1.set_ylabel('과적합 정도 (%)')
    ax1.set_title('과적합 정도 비교\n(훈련 정확도 - 테스트 정확도)')
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='주의선 (5%)')
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='위험선 (10%)')
    ax1.legend()
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. 과적합 vs 테스트 성능 산점도
    scatter = ax2.scatter(overfitting_scores, test_accuracies, 
                         s=100, c=range(len(models)), cmap='tab10', alpha=0.7)
    
    for i, model in enumerate(models):
        ax2.annotate(model, (overfitting_scores[i], test_accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('과적합 정도 (%)')
    ax2.set_ylabel('테스트 정확도 (%)')
    ax2.set_title('과적합 vs 테스트 성능')
    ax2.grid(True, alpha=0.3)
    
    # 이상적인 영역 표시 (낮은 과적합, 높은 성능)
    ax2.axvline(x=5, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=95, color='green', linestyle='--', alpha=0.5)
    ax2.text(1, 98, '이상적 영역\n(낮은 과적합,\n높은 성능)', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/images/Overall_Analysis/과적합_분석.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def create_efficiency_radar():
    """효율성 레이더 차트"""
    data = load_performance_data()
    
    # 정규화를 위한 최대/최소값 계산
    metrics = ['test_accuracy', 'training_time', 'prediction_time', 'memory_used']
    normalized_data = {}
    
    for model_name, model_data in data.items():
        normalized_data[model_name] = {}
        
        # 정확도는 높을수록 좋음 (그대로 유지)
        normalized_data[model_name]['accuracy'] = model_data['test_accuracy'] * 100
        
        # 시간과 메모리는 낮을수록 좋음 (역수 취해서 정규화)
        # 최대값으로 나누어 0-1 범위로 만든 후 1에서 빼서 역전
        max_train_time = max([d['training_time'] for d in data.values()])
        max_pred_time = max([d['prediction_time'] for d in data.values()])
        max_memory = max([abs(d['memory_used']) for d in data.values()])
        
        normalized_data[model_name]['speed'] = (1 - model_data['training_time'] / max_train_time) * 100
        normalized_data[model_name]['efficiency'] = (1 - model_data['prediction_time'] / max_pred_time) * 100
        normalized_data[model_name]['memory'] = (1 - abs(model_data['memory_used']) / max_memory) * 100
    
    # 상위 4개 모델만 선택 (차트 가독성을 위해)
    top_models = sorted(data.items(), 
                       key=lambda x: x[1]['test_accuracy'], 
                       reverse=True)[:4]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['정확도', '훈련 속도', '예측 효율성', '메모리 효율성']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 폐곡선을 위해 첫 번째 각도 추가
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (model_name, _) in enumerate(top_models):
        values = [
            normalized_data[model_name]['accuracy'],
            normalized_data[model_name]['speed'],
            normalized_data[model_name]['efficiency'],
            normalized_data[model_name]['memory']
        ]
        values += values[:1]  # 폐곡선을 위해 첫 번째 값 추가
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=model_name, color=colors[i], alpha=0.8)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title('상위 모델 종합 성능 레이더 차트\n(높을수록 좋음)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig('results/images/Overall_Analysis/효율성_레이더_차트.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("모델 성능 비교 차트 생성 중...")
    
    # 1. 종합 성능 비교
    create_performance_comparison()
    print("✅ 종합 성능 비교 차트 생성 완료!")
    
    # 2. 과적합 분석
    create_overfitting_analysis()
    print("✅ 과적합 분석 차트 생성 완료!")
    
    # 3. 효율성 레이더 차트
    create_efficiency_radar()
    print("✅ 효율성 레이더 차트 생성 완료!")
    
    print("\n모든 차트가 results/images/Overall_Analysis/ 폴더에 저장되었습니다!") 