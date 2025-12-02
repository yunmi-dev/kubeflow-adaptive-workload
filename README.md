# Kubeflow Adaptive Workload Distribution

## 프로젝트 개요
이기종 하드웨어 환경에서 각 노드의 실제 성능을 측정하여 머신러닝 학습 데이터를 차등 분배함으로써 전체 학습 시간을 단축하는 시스템

## 연구자
- **이름:** 정윤미
- **학번:** 2022105745
- **소속:** 경희대학교 컴퓨터공학과
- **지도교수:** 허의남 교수님

## 주요 기능
- 노드 성능 자동 측정 (CPU, Memory, Network, Load)
- 3가지 가중치 최적화 기법 비교
  - Grid Search
  - Random Search
  - Bayesian Optimization
- 성능 기반 데이터 분배
- 분산 학습 실험 자동화

## 시스템 요구사항
- Kubernetes Cluster (Kind)
- Python 3.8+
- TensorFlow 2.x
- Docker Desktop

## 설치 방법
```bash
# 클러스터 생성
kind create cluster --name ml-cluster --config kind-config.yaml

# 의존성 설치
pip install -r requirements.txt
```

## 실험 실행
```bash
# 전체 실험 자동화
python run_experiments.py
```

## 논문 및 참고자료
- 중간보고서: 링크예정
- 최종보고서: 링크예정

## License
MIT License