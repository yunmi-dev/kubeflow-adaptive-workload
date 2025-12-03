"""
분산 학습 시뮬레이터
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple

class DistributedTrainer:
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.batch_size = model_config.get('batch_size', 32)
        self.epochs = model_config.get('epochs', 5)
    
    def create_model(self) -> keras.Model:
        """
        간단한 CNN 모델 생성 (MNIST용)
        """
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def simulate_node_training(
        self, 
        node_id: int,
        data_size: int, 
        cpu_limit: float
    ) -> Tuple[float, Dict]:
        """
        특정 노드의 학습 시뮬레이션
        
        Args:
            node_id: 노드 ID
            data_size: 할당된 데이터 개수
            cpu_limit: CPU 성능 제한 (1.0, 0.5, 0.25)
        
        Returns:
            training_time: 학습 시간 (초)
            metrics: 학습 메트릭
        """
        print(f"\n[Node {node_id}] 학습 시작 (데이터: {data_size}, CPU: {cpu_limit*100}%)")
        
        # 실제 데이터 로드 (MNIST)
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        x_train = x_train[:data_size].reshape(-1, 28, 28, 1) / 255.0
        y_train = y_train[:data_size]
        
        # 모델 생성
        model = self.create_model()
        
        # CPU 제한 시뮬레이션: 배치 크기 조정
        effective_batch_size = max(8, int(self.batch_size * cpu_limit))
        
        # 학습 시작
        start_time = time.time()
        
        history = model.fit(
            x_train, y_train,
            batch_size=effective_batch_size,
            epochs=self.epochs,
            verbose=0,
            validation_split=0.1
        )
        
        # 학습 시간 계산
        training_time = time.time() - start_time
        
        # 성능 평가
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        metrics = {
            'node_id': node_id,
            'data_size': data_size,
            'cpu_limit': cpu_limit,
            'training_time': training_time,
            'final_loss': final_loss,
            'final_accuracy': final_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'batch_size': effective_batch_size
        }
        
        print(f"[Node {node_id}] 완료 - 시간: {training_time:.2f}s, 정확도: {final_acc:.4f}")
        
        return training_time, metrics
    
    def calculate_jct(self, training_times: List[float]) -> float:
        """
        JCT (Job Completion Time) 계산
        = 가장 늦게 끝난 노드의 시간
        """
        return max(training_times)
    
    def calculate_time_balance(self, training_times: List[float]) -> Dict[str, float]:
        """
        시간 균형도 계산
        """
        times = np.array(training_times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # 시간 균형도: 1 - (σ/T̄)
        balance = 1 - (std_time / mean_time) if mean_time > 0 else 0
        
        return {
            'mean_time': mean_time,
            'std_time': std_time,
            'balance': balance,
            'min_time': np.min(times),
            'max_time': np.max(times)
        }

if __name__ == "__main__":
    # 테스트
    config = {
        'batch_size': 32,
        'epochs': 2  # 테스트용 2 에폭
    }
    
    trainer = DistributedTrainer(config)
    
    print("=== 분산 학습 시뮬레이션 테스트 ===")
    
    # 3개 노드 시뮬레이션
    node_configs = [
        (1, 20000, 1.0),   # Node 1: 20k data, 100% CPU
        (2, 20000, 0.5),   # Node 2: 20k data, 50% CPU
        (3, 20000, 0.25),  # Node 3: 20k data, 25% CPU
    ]
    
    training_times = []
    all_metrics = []
    
    for node_id, data_size, cpu_limit in node_configs:
        train_time, metrics = trainer.simulate_node_training(node_id, data_size, cpu_limit)
        training_times.append(train_time)
        all_metrics.append(metrics)
    
    # JCT 계산
    jct = trainer.calculate_jct(training_times)
    balance_metrics = trainer.calculate_time_balance(training_times)
    
    print(f"\n=== 결과 ===")
    print(f"JCT (Job Completion Time): {jct:.2f}s")
    print(f"평균 시간: {balance_metrics['mean_time']:.2f}s")
    print(f"시간 편차: {balance_metrics['std_time']:.2f}s")
    print(f"시간 균형도: {balance_metrics['balance']:.4f}")