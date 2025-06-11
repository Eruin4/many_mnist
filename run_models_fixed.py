#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
수정된 모든 MNIST 분류 모델 실행 스크립트
exec() 대신 함수 호출 방식으로 안전하게 실행
"""

import os
import json
import time
import traceback
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.getcwd())

def run_all_models_safely():
    """모든 모델을 안전하게 실행"""
    print("=" * 80)
    print("MNIST 분류 모델 종합 실행 (완료된 프로젝트)")
    print("=" * 80)
    
    print("⚠️ 주의: 개별 모델 파일들은 이미 실행 완료되어 정리되었습니다.")
    print("현재 사용 가능한 기능:")
    print("1. 성능 비교 차트 생성")
    print("2. 결과 분석 보고서 확인")
    print("3. 정리된 결과 파일 확인")
    print("-" * 80)
    
    # 현재 사용 가능한 기능들
    available_functions = [
        ("성능 비교 차트", "create_performance_chart", "create_charts"),
        ("결과 요약", "결과 요약", "show_results_summary")
    ]
    
    results = {}
    total_start_time = time.time()
    
    # 1. 성능 비교 차트 생성
    print("\n📊 성능 비교 차트 생성 중...")
    try:
        import create_performance_chart
        
        print("종합 성능 비교 차트 생성...")
        create_performance_chart.create_performance_comparison()
        
        print("과적합 분석 차트 생성...")
        create_performance_chart.create_overfitting_analysis()
        
        print("효율성 레이더 차트 생성...")
        create_performance_chart.create_efficiency_radar()
        
        print("✅ 모든 성능 비교 차트가 생성되었습니다!")
        results['성능 차트'] = {'status': 'success'}
        
    except Exception as e:
        print(f"❌ 성능 차트 생성 실패: {str(e)}")
        results['성능 차트'] = {'status': 'error', 'error': str(e)}
    
    # 2. 결과 요약 표시
    show_results_summary()
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"작업 완료! 총 소요 시간: {total_time:.2f}초")
    print(f"{'='*80}")
    
    return results

def show_results_summary():
    """결과 요약 표시"""
    print("\n" + "="*60)
    print("📊 MNIST 모델 분석 결과 요약")
    print("="*60)
    
    try:
        metrics_file = "results/performance_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 정확도 순으로 정렬
            sorted_models = sorted(data.items(), 
                                 key=lambda x: x[1].get('test_accuracy', 0), 
                                 reverse=True)
            
            print("\n🏆 성능 순위 (테스트 정확도):")
            for i, (model, metrics) in enumerate(sorted_models, 1):
                acc = metrics.get('test_accuracy', 0) * 100
                time_val = metrics.get('training_time', 0)
                memory = abs(metrics.get('memory_used', 0))
                print(f"{i}위. {model:15s}: {acc:5.1f}% (훈련: {time_val:6.1f}초, 메모리: {memory:5.1f}MB)")
            
            print(f"\n📁 저장된 결과:")
            print(f"- 종합 분석 보고서: 종합_분석_보고서.md")
            print(f"- 시각화 결과: results/images/ (모델별 폴더)")
            print(f"- 분류 보고서: results/json/ (모델별 폴더)")
            print(f"- 성능 메트릭: results/performance_metrics.json")
            
            # 폴더 구조 확인
            if os.path.exists("results/images"):
                image_folders = [f for f in os.listdir("results/images") if os.path.isdir(f"results/images/{f}")]
                print(f"- 총 {len(image_folders)}개 모델 분석 완료")
        else:
            print("❌ 성능 메트릭 파일을 찾을 수 없습니다.")
            print("먼저 모델들을 실행해주세요.")
    
    except Exception as e:
        print(f"❌ 결과 요약 로드 실패: {e}")

def check_project_status():
    """프로젝트 상태 확인"""
    print("📋 프로젝트 상태 확인 중...")
    
    # 필수 파일들 확인
    essential_files = [
        "results/performance_metrics.json",
        "종합_분석_보고서.md",
        "create_performance_chart.py"
    ]
    
    essential_dirs = [
        "results/images",
        "results/json"
    ]
    
    print("\n📁 필수 파일 확인:")
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (누락)")
    
    print("\n📂 필수 폴더 확인:")
    for dir_path in essential_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (누락)")
    
    # 결과 폴더 구조 확인
    if os.path.exists("results/images"):
        image_folders = [f for f in os.listdir("results/images") if os.path.isdir(f"results/images/{f}")]
        print(f"\n📊 분석 완료된 모델: {len(image_folders)}개")
        for folder in sorted(image_folders):
            file_count = len([f for f in os.listdir(f"results/images/{folder}") if f.endswith('.png')])
            print(f"   - {folder}: {file_count}개 이미지")
    
    return True

if __name__ == "__main__":
    print("🚀 MNIST 프로젝트 실행기 시작...")
    print("현재 작업 디렉토리:", os.getcwd())
    
    # 1단계: 프로젝트 상태 확인
    print("\n1단계: 프로젝트 상태 확인")
    check_project_status()
    
    # 2단계: 사용 가능한 기능 실행
    print("\n2단계: 차트 생성 및 결과 요약")
    results = run_all_models_safely()
    
    print("\n" + "="*80)
    print("🎉 프로젝트 실행 완료!")
    print("📖 자세한 분석은 '종합_분석_보고서.md'를 확인하세요.")
    print("📊 시각화 결과는 'results/images/' 폴더를 확인하세요.")
    print("="*80) 