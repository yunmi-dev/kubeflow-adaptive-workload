"""
노드 성능 측정 및 점수 계산 모듈
"""
import time
import numpy as np
import psutil
from typing import Dict, Tuple

class PerformanceScorer:
    def __init__(self):
        self.baseline_flops = 1e9  # 1 GFLOPS 기준
        
    def measure_cpu_performance(self, matrix_size: int = 1000, iterations: int = 5) -> float:
        """
        CPU 성능 측정 (FLOPS 기반 행렬 연산)
        Returns: 정규화된 CPU 성능 점수 (0~1)
        """
        times = []
        for _ in range(iterations):
            A = np.random.rand(matrix_size, matrix_size)
            B = np.random.rand(matrix_size, matrix_size)
            
            start = time.time()
            C = np.dot(A, B)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        # FLOPS 계산: 2 * n^3 operations
        flops = (2 * matrix_size ** 3) / avg_time
        
        # 정규화 (0~1)
        normalized_score = min(flops / (10 * self.baseline_flops), 1.0)
        
        return normalized_score
    
    def measure_memory(self) -> float:
        """
        사용 가능한 메모리 비율 측정
        Returns: 메모리 점수 (0~1)
        """
        memory = psutil.virtual_memory()
        available_ratio = memory.available / memory.total
        return available_ratio
    
    def measure_network_bandwidth(self) -> float:
        """
        네트워크 대역폭 측정 (단순화: 임의값 반환)
        실제로는 iperf3 사용해야 하지만 시뮬레이션용
        Returns: 정규화된 네트워크 점수 (0~1)
        """
        # 시뮬레이션: 0.5~1.0 사이 랜덤값
        return 0.5 + np.random.rand() * 0.5
    
    def measure_load(self) -> float:
        """
        현재 CPU 부하율 측정
        Returns: 부하율 (0~1)
        """
        cpu_percent = psutil.cpu_percent(interval=1) / 100.0
        return cpu_percent
    
    def calculate_performance_score(
        self, 
        alpha: float, 
        beta: float, 
        gamma: float, 
        delta: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        성능 점수 계산
        S_i = α·CPU_i + β·Mem_i + γ·NetBW_i - δ·Load_i
        
        Returns:
            score: 성능 점수
            metrics: 각 메트릭 값들
        """
        cpu = self.measure_cpu_performance()
        mem = self.measure_memory()
        net = self.measure_network_bandwidth()
        load = self.measure_load()
        
        score = alpha * cpu + beta * mem + gamma * net - delta * load
        
        metrics = {
            'cpu': cpu,
            'memory': mem,
            'network': net,
            'load': load,
            'score': score
        }
        
        return score, metrics

    def simulate_node_performance(self, cpu_limit: float) -> Tuple[float, Dict[str, float]]:
        """
        특정 CPU 제한을 가진 노드의 성능 시뮬레이션
        cpu_limit: 1.0 (제한없음), 0.5 (50%), 0.25 (25%)
        """
        # CPU 성능은 제한에 비례
        cpu = cpu_limit * np.random.uniform(0.9, 1.0)
        
        # 메모리는 CPU와 무관하게 높은 값
        mem = np.random.uniform(0.7, 0.9)
        
        # 네트워크는 모든 노드 비슷
        net = np.random.uniform(0.8, 0.95)
        
        # 부하는 CPU 성능에 반비례 (느린 노드가 더 부하 높음)
        load = (1.0 - cpu_limit) * np.random.uniform(0.1, 0.3)
        
        metrics = {
            'cpu': cpu,
            'memory': mem,
            'network': net,
            'load': load
        }
        
        return metrics

if __name__ == "__main__":
    scorer = PerformanceScorer()
    print("=== 성능 측정 테스트 ===")
    
    # 기본 가중치로 테스트
    alpha, beta, gamma, delta = 0.4, 0.3, 0.2, 0.1
    score, metrics = scorer.calculate_performance_score(alpha, beta, gamma, delta)
    
    print(f"\n측정 결과:")
    print(f"  CPU: {metrics['cpu']:.4f}")
    print(f"  Memory: {metrics['memory']:.4f}")
    print(f"  Network: {metrics['network']:.4f}")
    print(f"  Load: {metrics['load']:.4f}")
    print(f"\n성능 점수: {score:.4f}")