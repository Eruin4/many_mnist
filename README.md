# MNIST 숫자 분류 프로젝트

이 프로젝트는 MNIST 손글씨 숫자 데이터셋을 다양한 머신러닝 및 딥러닝 알고리즘으로 분류하는 8개의 모듈로 구성되어 있습니다.

## 📁 프로젝트 구조

```
project-mnist/
├── create_performance_chart.py  # 성능 비교 차트 생성
├── run_models_fixed.py         # 모든 모델 실행 스크립트
├── requirements.txt            # 필요 라이브러리 목록
├── README.md                   # 프로젝트 설명서
├── 종합_분석_보고서.md           # 최종 분석 보고서
├── results/                    # 실행 결과 저장
│   ├── performance_metrics.json # 전체 성능 통계
│   ├── images/                 # 시각화 결과
│   │   ├── Overall_Analysis/   # 종합 분석 차트
│   │   ├── CNN/               # CNN 모델 분석
│   │   ├── DecisionTree/      # 의사결정트리 분석
│   │   ├── RandomForest/      # 랜덤포레스트 분석
│   │   ├── SVM/              # SVM 분석
│   │   ├── LogisticRegression/ # 로지스틱 회귀 분석
│   │   ├── KNN/              # KNN 분석
│   │   ├── XGBoost/          # XGBoost 분석
│   │   └── LightGBM/         # LightGBM 분석
│   └── json/                  # 분류 보고서
│       ├── CNN/              # CNN 분류 보고서
│       ├── DecisionTree/     # 의사결정트리 분류 보고서
│       ├── RandomForest/     # 랜덤포레스트 분류 보고서
│       ├── SVM/             # SVM 분류 보고서
│       ├── LogisticRegression/ # 로지스틱 회귀 분류 보고서
│       ├── KNN/             # KNN 분류 보고서
│       ├── XGBoost/         # XGBoost 분류 보고서
│       └── LightGBM/        # LightGBM 분류 보고서
└── 개별 모델 파일들 (실행 완료 후 정리됨)
```

## 🚀 설치 및 실행

### 1. 가상환경 활성화
```bash
# Windows
project\ai_analysis_env\Scripts\activate

# Linux/Mac
source project/ai_analysis_env/bin/activate
```

### 2. 필요 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. 실행 방법

#### 모든 모델 한번에 실행 (권장)
```bash
python run_models_fixed.py
```

#### 성능 비교 차트 생성
```bash
python create_performance_chart.py
```

#### 개별 모델 분석 확인
실행이 완료되면 `results/` 폴더에서 다음과 같은 분석 결과를 확인할 수 있습니다:
- 시각화 차트: `results/images/모델명/`
- 분류 보고서: `results/json/모델명/`
- 종합 분석: `종합_분석_보고서.md`

## 📊 각 모델별 특징

### 1. 의사결정나무 (Decision Tree)
- **특징**: 해석이 쉬운 트리 구조 모델
- **장점**: 직관적이고 설명 가능
- **분석 내용**: 특성 중요도, 트리 깊이, 혼동 행렬

### 2. 랜덤포레스트 (Random Forest)
- **특징**: 여러 의사결정나무의 앙상블
- **장점**: 과적합 방지, 높은 성능
- **분석 내용**: 특성 중요도, 트리 통계, OOB 점수

### 3. SVM (Support Vector Machine)
- **특징**: 마진을 최대화하는 분류기
- **장점**: 고차원 데이터에 효과적
- **분석 내용**: 서포트 벡터, 커널 비교, 결정 경계

### 4. 로지스틱 회귀 (Logistic Regression)
- **특징**: 확률 기반 선형 분류기
- **장점**: 빠르고 해석 가능
- **분석 내용**: 회귀 계수, 정규화 분석, 예측 신뢰도

### 5. KNN (K-Nearest Neighbors)
- **특징**: 게으른 학습 알고리즘
- **장점**: 단순하고 직관적
- **분석 내용**: 최적 K 값, 거리 메트릭 비교, 이웃 분석

### 6. XGBoost
- **특징**: 그래디언트 부스팅 알고리즘
- **장점**: 높은 성능, 특성 중요도 제공
- **분석 내용**: 하이퍼파라미터 튜닝, 학습 곡선, 트리 개수 분석

### 7. LightGBM
- **특징**: 효율적인 그래디언트 부스팅
- **장점**: 빠른 훈련 속도, 메모리 효율성
- **분석 내용**: 부스팅 타입 비교, 잎 노드 분석, 훈련 진행상황

### 8. CNN (Convolutional Neural Network)
- **특징**: 이미지 처리에 특화된 신경망
- **장점**: 공간적 특성 추출, 높은 성능
- **분석 내용**: 컨볼루션 필터, 특성 맵, 아키텍처 비교

## 📈 결과 비교

### 최종 성능 순위 (테스트 정확도 기준)
1. **CNN**: 98.15% ⭐ (최고 성능)
2. **SVM**: 95.85%
3. **KNN**: 94.95%
4. **LightGBM**: 94.30%
5. **Random Forest**: 93.90%
6. **XGBoost**: 93.75%
7. **Logistic Regression**: 90.45%
8. **Decision Tree**: 77.55%

### 평가 메트릭
각 모델은 다음과 같은 메트릭으로 평가됩니다:
- **정확도 (Accuracy)**
- **정밀도 (Precision)**
- **재현율 (Recall)**
- **F1-점수 (F1-Score)**
- **혼동 행렬 (Confusion Matrix)**
- **훈련 시간 및 예측 시간**
- **메모리 사용량**

## 🔧 하이퍼파라미터 튜닝

대부분의 모델에 하이퍼파라미터 튜닝이 포함되어 있습니다:
- Grid Search
- Cross Validation
- Validation Curve 분석

## 📊 시각화

### 종합 분석 차트 (`results/images/Overall_Analysis/`)
- 모델 종합 성능 비교 (6개 서브플롯)
- 과적합 분석 차트
- 효율성 레이더 차트

### 모델별 특화 시각화 (`results/images/모델명/`)
각 모델은 다양한 시각화를 제공합니다:
- 혼동 행렬 히트맵
- 특성 중요도 차트 (해당 모델)
- 학습 곡선
- 하이퍼파라미터 튜닝 결과
- 모델별 특화 분석 (서포트 벡터, CNN 필터 등)

### 체계적 결과 정리
- **총 58개 분석 이미지**: 모델별로 폴더 정리
- **18개 JSON 보고서**: 상세한 분류 성능 데이터
- **종합 분석 보고서**: 모든 모델 비교 및 권장사항

## ⚠️ 주의사항

1. **데이터 크기**: 실행 시간을 고려하여 데이터 크기를 축소했습니다.
2. **라이브러리 의존성**: 일부 모델은 추가 라이브러리가 필요할 수 있습니다.
3. **GPU 지원**: CNN 모델은 GPU 사용 시 더 빠른 훈련이 가능합니다.

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 👨‍💻 사용 환경

- **Python**: 3.8+
- **가상환경**: ai_analysis_env
- **운영체제**: Windows, Linux, macOS

## 📚 추가 자료

- [MNIST 데이터셋 정보](http://yann.lecun.com/exdb/mnist/)
- [Scikit-learn 문서](https://scikit-learn.org/)
- [TensorFlow 문서](https://www.tensorflow.org/)
- [XGBoost 문서](https://xgboost.readthedocs.io/)
- [LightGBM 문서](https://lightgbm.readthedocs.io/) 