# MNIST 분류 모델 종합 분석 보고서

실행 일시: 2025-06-11 19:48:45
데이터셋 크기: 10,000개 (훈련: 8,000개, 테스트: 2,000개)

## 1. 성능 순위

| 순위 | 모델 | 테스트 정확도 | 훈련 시간 | 메모리 사용량 |
|------|------|---------------|-----------|---------------|
| 1 | SVM | 0.9585 | 157.86s | 154.5MB |
| 2 | KNN | 0.9495 | 14.93s | 91.8MB |
| 3 | RandomForest | 0.9390 | 21.55s | 205.6MB |
| 4 | LogisticRegression | 0.9045 | 41.84s | 156.5MB |
| 5 | DecisionTree | 0.7755 | 7.98s | 100.5MB |
| 6 | CNN | 0.1945 | 18.89s | 508.5MB |

## 2. 분석 결과

### 최고 정확도: SVM
- 테스트 정확도: 0.9585
- 훈련 시간: 157.86초
- 메모리 사용량: 154.5MB

### 최고 속도: DecisionTree
- 훈련 시간: 7.98초
- 테스트 정확도: 0.7755

### 최고 메모리 효율성: KNN
- 메모리 사용량: 91.8MB
- 테스트 정확도: 0.9495

## 3. 권장사항

- **정확도 우선**: CNN 또는 XGBoost 권장
- **속도 우선**: 의사결정나무 또는 로지스틱 회귀 권장
- **메모리 효율성**: KNN 또는 의사결정나무 권장
- **균형잡힌 성능**: 랜덤포레스트 또는 SVM 권장

## 4. 생성된 시각화 파일

- CNN_Basic_confusion_matrix.png
- CNN_Basic_training_history.png
- CNN_Improved_confusion_matrix.png
- CNN_Improved_training_history.png
- CNN_class_performance.png
- CNN_filters.png
- CNN_predictions.png
- DecisionTree_Basic_confusion_matrix.png
- DecisionTree_Optimized_confusion_matrix.png
- DecisionTree_class_performance.png
- DecisionTree_feature_importance.png
- DecisionTree_importance_heatmap.png
- DecisionTree_learning_curve_max_depth.png
- KNN_Basic_confusion_matrix.png
- KNN_Optimized_confusion_matrix.png
- KNN_class_performance.png
- KNN_distance_distribution.png
- KNN_learning_curve_n_neighbors.png
- KNN_neighbor_analysis.png
- LogisticRegression_Basic_confusion_matrix.png
- LogisticRegression_Optimized_confusion_matrix.png
- LogisticRegression_class_performance.png
- LogisticRegression_coefficient_norms.png
- LogisticRegression_coefficients_class0.png
- LogisticRegression_learning_curve_C.png
- LogisticRegression_probability_analysis.png
- RandomForest_Basic_confusion_matrix.png
- RandomForest_Optimized_confusion_matrix.png
- RandomForest_class_performance.png
- RandomForest_feature_importance.png
- RandomForest_importance_heatmap.png
- RandomForest_importance_variance.png
- RandomForest_learning_curve_n_estimators.png
- SVM_Linear_confusion_matrix.png
- SVM_Optimized_confusion_matrix.png
- SVM_RBF_confusion_matrix.png
- SVM_class_performance.png
- SVM_decision_function.png
- SVM_learning_curve_C.png
- SVM_learning_curve_gamma.png
- SVM_support_vectors.png
- accuracy_comparison.png
- comprehensive_comparison.png
- memory_comparison.png
- performance_radar_chart.png
- time_comparison.png
