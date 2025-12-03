"""
가중치 최적화 모듈
- Grid Search
- Random Search
- Bayesian Optimization
"""
import numpy as np
import time
from typing import List, Dict, Tuple, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
import itertools

class WeightOptimizer:
    def __init__(self, objective_function: Callable):
        """
        Args:
            objective_function: JCT를 반환하는 함수
                signature: f(alpha, beta, gamma, delta) -> float
        """
        self.objective_function = objective_function
        self.history = []
    
    def grid_search(
        self,
        alpha_range: List[float] = [0.3, 0.35, 0.4, 0.45, 0.5],
        beta_range: List[float] = [0.2, 0.25, 0.3],
        gamma_range: List[float] = [0.1, 0.15, 0.2],
        delta_constraint: Tuple[float, float] = (0.1, 0.2)
    ) -> Dict:
        """
        Grid Search 최적화
        
        Returns:
            최적 결과 및 탐색 히스토리
        """
        print("="*60)
        print("Grid Search 시작")
        print(f"탐색 공간: {len(alpha_range)} × {len(beta_range)} × {len(gamma_range)} = {len(alpha_range)*len(beta_range)*len(gamma_range)} 조합")
        print("="*60)
        
        results = []
        start_time = time.time()
        
        total_combinations = len(alpha_range) * len(beta_range) * len(gamma_range)
        count = 0
        
        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    delta = 1.0 - alpha - beta - gamma
                    
                    # delta 제약 조건 확인
                    if not (delta_constraint[0] <= delta <= delta_constraint[1]):
                        continue
                    
                    count += 1
                    print(f"\n[{count}/{total_combinations}] α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}, δ={delta:.2f}")
                    
                    # JCT 계산
                    jct = self.objective_function(alpha, beta, gamma, delta)
                    
                    result = {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'delta': delta,
                        'jct': jct
                    }
                    results.append(result)
                    self.history.append(result)
                    
                    print(f"  → JCT: {jct:.2f}s")
        
        elapsed = time.time() - start_time
        
        # 최적 결과 찾기
        best = min(results, key=lambda x: x['jct'])
        
        print(f"\n{'='*60}")
        print("Grid Search 완료")
        print(f"실행 시간: {elapsed:.2f}s")
        print(f"총 탐색: {len(results)}개 조합")
        print(f"\n최적 가중치:")
        print(f"  α={best['alpha']:.2f}, β={best['beta']:.2f}, γ={best['gamma']:.2f}, δ={best['delta']:.2f}")
        print(f"  JCT: {best['jct']:.2f}s")
        print("="*60)
        
        return {
            'method': 'Grid Search',
            'best_weights': best,
            'all_results': results,
            'elapsed_time': elapsed,
            'num_evaluations': len(results)
        }
    
    def random_search(
        self,
        n_iterations: int = 100,
        alpha_range: Tuple[float, float] = (0.3, 0.5),
        beta_range: Tuple[float, float] = (0.2, 0.3),
        gamma_range: Tuple[float, float] = (0.1, 0.2),
        delta_constraint: Tuple[float, float] = (0.1, 0.2)
    ) -> Dict:
        """
        Random Search 최적화
        
        Returns:
            최적 결과 및 탐색 히스토리
        """
        print("="*60)
        print(f"Random Search 시작 ({n_iterations}회 샘플링)")
        print("="*60)
        
        results = []
        start_time = time.time()
        
        valid_count = 0
        attempts = 0
        max_attempts = n_iterations * 10
        
        while valid_count < n_iterations and attempts < max_attempts:
            attempts += 1
            
            # 랜덤 샘플링
            alpha = np.random.uniform(*alpha_range)
            beta = np.random.uniform(*beta_range)
            gamma = np.random.uniform(*gamma_range)
            delta = 1.0 - alpha - beta - gamma
            
            # delta 제약 조건 확인
            if not (delta_constraint[0] <= delta <= delta_constraint[1]):
                continue
            
            valid_count += 1
            
            if valid_count % 10 == 0:
                print(f"\n[{valid_count}/{n_iterations}] α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")
            
            # JCT 계산
            jct = self.objective_function(alpha, beta, gamma, delta)
            
            result = {
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'delta': delta,
                'jct': jct
            }
            results.append(result)
            self.history.append(result)
            
            if valid_count % 10 == 0:
                print(f"  → JCT: {jct:.2f}s")
        
        elapsed = time.time() - start_time
        
        # 최적 결과
        best = min(results, key=lambda x: x['jct'])
        
        print(f"\n{'='*60}")
        print("Random Search 완료")
        print(f"실행 시간: {elapsed:.2f}s")
        print(f"총 샘플링: {len(results)}개")
        print(f"\n최적 가중치:")
        print(f"  α={best['alpha']:.3f}, β={best['beta']:.3f}, γ={best['gamma']:.3f}, δ={best['delta']:.3f}")
        print(f"  JCT: {best['jct']:.2f}s")
        print("="*60)
        
        return {
            'method': 'Random Search',
            'best_weights': best,
            'all_results': results,
            'elapsed_time': elapsed,
            'num_evaluations': len(results)
        }
    
    def bayesian_optimization(
        self,
        n_iterations: int = 30,
        n_initial: int = 10,
        alpha_range: Tuple[float, float] = (0.3, 0.5),
        beta_range: Tuple[float, float] = (0.2, 0.3),
        gamma_range: Tuple[float, float] = (0.1, 0.2),
        delta_constraint: Tuple[float, float] = (0.1, 0.2)
    ) -> Dict:
        """
        Bayesian Optimization 최적화
        
        Returns:
            최적 결과 및 탐색 히스토리
        """
        print("="*60)
        print(f"Bayesian Optimization 시작 ({n_iterations}회)")
        print("="*60)
        
        results = []
        start_time = time.time()
        
        # 1. 초기 랜덤 샘플링
        print(f"\n[초기화] {n_initial}개 랜덤 샘플링...")
        X_init = []
        y_init = []
        
        for i in range(n_initial * 3):  # 제약 조건 때문에 여유있게
            alpha = np.random.uniform(*alpha_range)
            beta = np.random.uniform(*beta_range)
            gamma = np.random.uniform(*gamma_range)
            delta = 1.0 - alpha - beta - gamma
            
            if not (delta_constraint[0] <= delta <= delta_constraint[1]):
                continue
            
            if len(X_init) >= n_initial:
                break
            
            X_init.append([alpha, beta, gamma])
            jct = self.objective_function(alpha, beta, gamma, delta)
            y_init.append(jct)
            
            result = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta, 'jct': jct}
            results.append(result)
            self.history.append(result)
            
            print(f"  {i+1}/{n_initial}: JCT={jct:.2f}s")
        
        X_observed = np.array(X_init)
        y_observed = np.array(y_init)
        
        # 2. Gaussian Process 모델
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        
        # 3. 반복 최적화
        print(f"\n[Bayesian 탐색] {n_iterations - n_initial}회...")
        
        for iteration in range(n_initial, n_iterations):
            # GP 학습
            gp.fit(X_observed, y_observed)
            
            # Acquisition Function 최대화 (Expected Improvement)
            best_y = np.min(y_observed)
            
            def neg_ei(x):
                x = np.array(x).reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                
                with np.errstate(divide='warn'):
                    imp = best_y - mu
                    Z = imp / sigma
                    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                    ei[sigma == 0.0] = 0.0
                
                return -ei[0]
            
            # 다음 샘플링 포인트 찾기 (여러 번 시도)
            best_x = None
            best_ei_val = float('inf')
            
            for _ in range(20):
                x0 = [
                    np.random.uniform(*alpha_range),
                    np.random.uniform(*beta_range),
                    np.random.uniform(*gamma_range)
                ]
                
                bounds = [alpha_range, beta_range, gamma_range]
                
                res = minimize(neg_ei, x0, bounds=bounds, method='L-BFGS-B')
                
                if res.fun < best_ei_val:
                    alpha, beta, gamma = res.x
                    delta = 1.0 - alpha - beta - gamma
                    
                    if delta_constraint[0] <= delta <= delta_constraint[1]:
                        best_x = res.x
                        best_ei_val = res.fun
            
            if best_x is None:
                # fallback: 랜덤 샘플링
                for _ in range(50):
                    alpha = np.random.uniform(*alpha_range)
                    beta = np.random.uniform(*beta_range)
                    gamma = np.random.uniform(*gamma_range)
                    delta = 1.0 - alpha - beta - gamma
                    
                    if delta_constraint[0] <= delta <= delta_constraint[1]:
                        best_x = [alpha, beta, gamma]
                        break
            
            alpha, beta, gamma = best_x
            delta = 1.0 - alpha - beta - gamma
            
            print(f"\n[{iteration+1}/{n_iterations}] α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")
            
            # 실제 평가
            jct = self.objective_function(alpha, beta, gamma, delta)
            
            result = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta, 'jct': jct}
            results.append(result)
            self.history.append(result)
            
            print(f"  → JCT: {jct:.2f}s (현재 최고: {np.min(y_observed):.2f}s)")
            
            # 관측 데이터 업데이트
            X_observed = np.vstack([X_observed, best_x])
            y_observed = np.append(y_observed, jct)
        
        elapsed = time.time() - start_time
        
        # 최적 결과
        best = min(results, key=lambda x: x['jct'])
        
        print(f"\n{'='*60}")
        print("Bayesian Optimization 완료")
        print(f"실행 시간: {elapsed:.2f}s")
        print(f"총 평가: {len(results)}회")
        print(f"\n최적 가중치:")
        print(f"  α={best['alpha']:.3f}, β={best['beta']:.3f}, γ={best['gamma']:.3f}, δ={best['delta']:.3f}")
        print(f"  JCT: {best['jct']:.2f}s")
        print("="*60)
        
        return {
            'method': 'Bayesian Optimization',
            'best_weights': best,
            'all_results': results,
            'elapsed_time': elapsed,
            'num_evaluations': len(results)
        }

if __name__ == "__main__":
    # 테스트용 목적 함수
    def dummy_objective(alpha, beta, gamma, delta):
        # 가상의 JCT 함수 (실제로는 분산 학습 실행)
        score = alpha * 1.0 + beta * 0.8 + gamma * 0.5 - delta * 0.3
        jct = 50 / score  # 점수가 높을수록 시간 짧음
        return jct + np.random.normal(0, 1)  # 노이즈 추가
    
    optimizer = WeightOptimizer(dummy_objective)
    
    print("Grid Search 테스트")
    grid_result = optimizer.grid_search(
        alpha_range=[0.4, 0.45],
        beta_range=[0.25, 0.3],
        gamma_range=[0.15, 0.2]
    )
    
    print(f"\n최적 결과: {grid_result['best_weights']}")