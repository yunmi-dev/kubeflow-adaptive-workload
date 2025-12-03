"""
노드 성능 측정 및 점수 계산 모듈 (이기종 환경 시뮬레이션)
"""
import time
import numpy as np
import psutil
from typing import Dict, Tuple

class PerformanceScorer:
    def __init__(self):
        self.baseline_flops = 1e9  # 1 GFLOPS 기준
        
    def measure_cpu_performance(self, matrix_size: int = 1000, iterations: int = 3) -> float:
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
        네트워크 대역폭 측정 (시뮬레이션)
        실제로는 iperf3 사용
        Returns: 정규화된 네트워크 점수 (0~1)
        """
        # 시뮬레이션: 0.7~0.95 사이
        return 0.7 + np.random.rand() * 0.25
    
    def measure_load(self) -> float:
        """
        현재 CPU 부하율 측정
        Returns: 부하율 (0~1)
        """
        cpu_percent = psutil.cpu_percent(interval=0.5) / 100.0
        return cpu_percent
    
    def simulate_node_metrics(self, cpu_factor: float) -> Dict[str, float]:
        """
        특정 CPU 성능을 가진 노드의 메트릭 시뮬레이션
        
        Args:
            cpu_factor: CPU 성능 계수 (1.0, 0.5, 0.25)
        
        Returns:
            각 메트릭 값
        """
        # CPU 성능은 factor에 비례
        cpu = cpu_factor * np.random.uniform(0.95, 1.0)
        
        # 메모리는 CPU와 어느 정도 상관관계 (성능 좋은 기기가 메모리도 많음)
        mem = 0.6 + cpu_factor * 0.3 + np.random.uniform(0, 0.1)
        mem = min(mem, 0.95)
        
        # 네트워크는 비교적 균등 (같은 네트워크 환경)
        net = np.random.uniform(0.8, 0.95)
        
        # 부하는 CPU 성능에 반비례 (느린 노드가 더 바쁨)
        base_load = 0.1
        load_penalty = (1.0 - cpu_factor) * 0.2
        load = base_load + load_penalty + np.random.uniform(0, 0.05)
        load = min(load, 0.4)
        
        metrics = {
            'cpu': cpu,
            'memory': mem,
            'network': net,
            'load': load
        }
        
        return metrics
    
    def calculate_performance_score(
        self, 
        metrics: Dict[str, float],
        alpha: float, 
        beta: float, 
        gamma: float, 
        delta: float
    ) -> float:
        """
        성능 점수 계산
        S_i = α·CPU_i + β·Mem_i + γ·NetBW_i - δ·Load_i
        
        Args:
            metrics: 노드 메트릭 (cpu, memory, network, load)
            alpha, beta, gamma, delta: 가중치
        
        Returns:
            성능 점수
        """
        score = (
            alpha * metrics['cpu'] + 
            beta * metrics['memory'] + 
            gamma * metrics['network'] - 
            delta * metrics['load']
        )
        
        return max(score, 0.01)  # 최소값 보장

if __name__ == "__main__":
    scorer = PerformanceScorer()
    print("=== 성능 측정 시뮬레이션 ===\n")
    
    # 3개 노드 시뮬레이션
    cpu_factors = [1.0, 0.5, 0.25]
    alpha, beta, gamma, delta = 0.4, 0.3, 0.2, 0.1
    
    for i, cpu_factor in enumerate(cpu_factors, 1):
        metrics = scorer.simulate_node_metrics(cpu_factor)
        score = scorer.calculate_performance_score(metrics, alpha, beta, gamma, delta)
        
        print(f"Node {i} (CPU {cpu_factor*100:.0f}%):")
        print(f"  CPU: {metrics['cpu']:.4f}")
        print(f"  Memory: {metrics['memory']:.4f}")
        print(f"  Network: {metrics['network']:.4f}")
        print(f"  Load: {metrics['load']:.4f}")
        print(f"  Score: {score:.4f}\n")