"""
성능 기반 데이터 분배 모듈
"""
import numpy as np
from typing import List, Dict

class DataDistributor:
    def __init__(self, total_data_size: int):
        self.total_data_size = total_data_size
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        성능 점수 정규화
        S'_i = S_i / Σ S_j
        """
        total = sum(scores)
        if total == 0:
            # 모든 점수가 0인 경우 균등 분배
            return [1.0 / len(scores)] * len(scores)
        
        normalized = [s / total for s in scores]
        return normalized
    
    def distribute_data(self, performance_scores: List[float]) -> List[int]:
        """
        성능 점수에 따라 데이터 분배
        Data_i = Total_Data × S'_i
        
        Args:
            performance_scores: 각 노드의 성능 점수
        
        Returns:
            각 노드에 할당할 데이터 개수 리스트
        """
        normalized_scores = self.normalize_scores(performance_scores)
        
        # 데이터 분배
        distributions = [int(self.total_data_size * score) for score in normalized_scores]
        
        # 반올림 오차 보정 (총합이 total_data_size가 되도록)
        diff = self.total_data_size - sum(distributions)
        if diff > 0:
            # 가장 큰 점수를 가진 노드에 추가
            max_idx = normalized_scores.index(max(normalized_scores))
            distributions[max_idx] += diff
        
        return distributions
    
    def get_baseline_distribution(self, num_nodes: int) -> List[int]:
        """
        균등 분배 (Baseline)
        """
        per_node = self.total_data_size // num_nodes
        distributions = [per_node] * num_nodes
        
        # 나머지 처리
        remainder = self.total_data_size % num_nodes
        for i in range(remainder):
            distributions[i] += 1
        
        return distributions

if __name__ == "__main__":
    # 테스트
    distributor = DataDistributor(total_data_size=60000)
    
    # 시뮬레이션: 3개 노드 (100%, 50%, 25% 성능)
    scores = [1.0, 0.5, 0.25]
    
    print("=== 데이터 분배 테스트 ===")
    print(f"총 데이터: {distributor.total_data_size}")
    print(f"성능 점수: {scores}")
    
    # Baseline (균등 분배)
    baseline = distributor.get_baseline_distribution(3)
    print(f"\nBaseline (균등): {baseline}")
    
    # 성능 기반 분배
    adaptive = distributor.distribute_data(scores)
    print(f"Adaptive (성능): {adaptive}")
    
    # 비율 확인
    print(f"\n분배 비율:")
    for i, (b, a) in enumerate(zip(baseline, adaptive)):
        print(f"  Node {i+1}: Baseline={b} ({b/60000*100:.1f}%), Adaptive={a} ({a/60000*100:.1f}%)")